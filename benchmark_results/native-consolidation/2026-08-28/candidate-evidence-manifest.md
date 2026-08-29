# Native Consolidation Candidate Evidence

Recorded August 28, 2026 on `picard-4090`. This is a compact, content-addressed
session manifest, not an accepted benchmark baseline. The benchmark/profile
artifacts identify base revision `78ec943445113fb8720a584ee7ef60a1814ace7a`
and dirty-worktree source digest
`ebb9ac6c6d3bc66a188f59eaec409c7d051985af21aaf48e076effd07a4986c7`.
Post-measurement correctness, provenance, and pickle-state fixes changed the
final session source digest to
`c8a58295928e5bc8bcf0623b45f86824ca0a8e4e35c64ab2631b37be26b210fd`.
The final tabular breadth checkpoint has current source digest
`729a464d74ef6785e785a8191e4d2e22c9786e3fb73d3ba7ef0ba8a0637671ae`.
The restricted query/eval and bounded merge/join checkpoint advances the
current source digest to
`9ff53ae6943e6be87b033575187068dc888eb40bed6150375f4e3f52229f41b5`.
The final Q6 capacity fix and same-source full pipeline advance it to
`b99fe3c6d0b36a66044c92e2c35437cfcb83b3b7a1e2aba12f8098e7bb0aef8b`.
The reviewed exact-compaction and two-phase memory-admission boundary advance
the final source digest to
`5f2f51b0ae52f3531a1150bcf7e0e194b5beb0f279ff47661d79e970d6f365d2`.
The broad-suite transfer audit and grouped-point negative-proof correction
advance the reviewed final source digest to
`1191b97172a662ccd2d5f9868d5c064d09a7d18ef7db9b4f9c94231bc4b75124`.
Separating the unfinished `build_area` and Q11 exhibit workstreams yields the
landing-tree runtime digest
`2107dee8d82084d086913376219ccfc1bfcbd992d9f9aef66a9193cc72ebbf93`.
Therefore only rows naming their digest are same-source evidence, and the clean
candidate must still be remeasured for final acceptance.

## Candidate Measurements

| Evidence | SHA-256 | Result |
|---|---|---|
| `/tmp/native-consolidation-current/core-10k-r3.json` | `ea3a8ddcc88c933719345b857220c291cf9780a5ae5cd746ef8bca3598782439` | 14/14 exact; GeoPandas 3.4681 s, vibeSpatial 2.7161 s; 1.277x aggregate, 1.004x geomean; 6/14 individual 10K losses. |
| `/tmp/native-consolidation-current/core-100k-r3.json` | `b1fda0eeb66720199493f5bea0320b40f7b8d3133115d110884af48c98a3c7a7` | 14/14 exact; GeoPandas 167.2154 s, vibeSpatial 13.4651 s; 12.418x aggregate, 4.989x geomean; habitat and insurance are the only losses. |
| `/tmp/native-consolidation-current/external-10k-r3.json` | `83bd6fe84c245fddaf4d5a27cb7061b08ada19bc18074ded9c63648795afbc1b` | 6/6 contracts pass; 1.139x aggregate, 1.029x geomean. |
| `/tmp/native-consolidation-current/external-100k-r3.json` | `a9a8091113b92c3ed7b679d7abc7dcc68252f76437a3091d3be06d1e2208f4dc` | 6/6 contracts pass; 6.871x aggregate, 1.576x geomean. |
| `/tmp/native-consolidation-current/external-1m-r3.json` | `85e632839900c0efcfba6c80f09d3e86d70d4952b0b130a97bbb4c599ed9446e` | 6/6 contracts pass; 6.136x aggregate, 1.735x geomean. Requested scale is asset-capped: only GeoLife reaches 1,000,000 input rows, so this does not satisfy the core 1M gate. |
| `/tmp/native-habitat-final/habitat-100k-page-formal-r3.json` | `148c195a4d56c8aa359513e10d771449aa4f949f8e9b502c4e880514819a9936` | Exact; 2.5081 s versus 2.4383 s GeoPandas, 0.972x; 332,788,564-byte peak device memory. |
| `/tmp/native-habitat-final/habitat-100k-page-profile.json` | `50fa4ef5d09f2bd623ad381df947fab38f8a0b54caae44a10b9e79cda0c060de` | Exact; 43/43 GPU steps, no fallback/offramp/H2D, 79 bounded D2H packets totaling 1,429,914 bytes and 1.87 ms. |
| `/tmp/native-consolidation-current/full-pipeline.json` | `4c9c133a777da35c5d7682d0f984982a7f69b765db3acf9ae5334875baf0901f` | 22 active results pass, two raster lanes deferred; zero compute materializations/fallbacks; 40 bounded planning packets, 41,056 bytes; slowest stage 77.15 ms. |
| `/tmp/native-consolidation-current/full-pipeline-final-source.json` | `e29d9e142176e08b76195dc19caf044480ad93ac8a8a83c6e02e677b9598a821` | Final session source digest; 22 active results pass, two raster lanes deferred; zero compute materializations/fallbacks; 40 bounded planning packets, 41,056 bytes; slowest stage 99.37 ms. |
| `/tmp/native-consolidation-current/full-pipeline-tabular-breadth-final.json` | `ff514b74d478bc6b86c2bf6c9769a84e9a891c74622f98b3f1e83ce97f8d1e4e` | Source digest `5aa4f7fb...`; 22 active 100K/1M results pass, two raster lanes deferred; zero fallback or compute materialization; 40 bounded planning packets/41,056 bytes; max 1M stage 73.56 ms. |
| `/tmp/native-consolidation-current/full-pipeline-topk-final.json` | `c19ca014d83e8e53e7b1e547f52adee2e718b6cae7afddfd1c90d0e46e3fb0a2` | Source digest `729a464d...`; 22 active 100K/1M results pass, two raster lanes deferred; zero fallback or compute materialization; 40 bounded planning packets/41,056 bytes; max 1M stage 75.42 ms. |
| `full-pipeline-query-merge-final.json` | `08c0063e74f4180f9a5cc7c5a4c98f832710897c2aed42d537ad95972b464d18` | Source digest `9ff53ae6...`; 22 active 100K/1M results pass, two raster lanes deferred; zero fallback or compute materialization; 40 bounded planning packets/41,056 bytes. All 51 one-million-row stages were reviewed; none exceeds 74.54 ms. |
| `topk-skew-scale-probe.json` | `fa27a0b659e3bb19252b8b5f7b2dccbe72973be8dfafead44c2c69365786b8e1` | Source digest `729a464d...`; exact constant-primary-key top 100 is 0.399x pandas at 10K, 0.840x at 100K, and 4.988x at 1M. This is a tabular crossover probe, not an accepted GeoPandas baseline. |
| `native-public-tabular-scale-probe.json` | `26855c4646e99bdd1fd1e076862e0dbe149d83cef49b32f610de64082e84e5a5` | Source digest `9ff53ae6...`; exact restricted query is 0.386x/0.710x/4.230x pandas and bounded unique-right merge is 0.521x/1.109x/6.903x at 10K/100K/1M. CPU wins the small shapes; merge crosses by 100K and both win at 1M. This is crossover evidence, not an accepted shootout baseline. |
| `core-10k-query-merge-profile.json` | `b4d9bd67d61488470c2773a93c86ef85e96b7c4ee51f30969f936c55bce30efb` | Source digest `9ff53ae6...`; 14/14 exact, zero fallback, no stage above 221 ms. GeoPandas 3.4681 s versus vibeSpatial 2.7116 s; 1.279x aggregate with six small-scale losses. The prior fully classified 66-event internal boundary ledger is unchanged. |
| `external-10k-query-merge-profile.json` | `b3283bf24f302c6e446759c0f03f54ea1e5beefbde2a3311d471b2ee9439612c` | Source digest `9ff53ae6...`; 6/6 exact, zero fallback; 1.143x aggregate and 1.035x geomean. Five small-scale losses remain, led by the asset-capped OSM enrichment control. |
| `external-100k-query-merge-profile.json` | `39e0d738e1527f8e98a1fc0bcc467d1485a2dad20393c0f4c21e0cf810eb6eb4` | Source digest `9ff53ae6...`; 6/6 exact, zero fallback; 7.016x aggregate and 1.606x geomean. CMAB is 26.95x and Power is 2.85x; four capped/small controls remain CPU-faster. |
| `external-1m-query-merge-profile.json` | `65316c5885d424af65cb34c54a35c06eab9cb32e43087fb8450259dfbff9e0fb` | Source digest `9ff53ae6...`; 6/6 exact, zero fallback; 6.134x aggregate and 1.745x geomean. Only GeoLife reaches 1M input rows; CMAB and Power are corpus-capped. |
| `full-pipeline-after-q6-fix.json` | `4a6c148a2f9585f4e3baaaf4e9a5a478f85e7a84f698cb7d67bd4a4ed2cc9680` | Source digest `b99fe3c6...`; 22/22 active 100K/1M lanes pass, two raster lanes deferred, zero fallback/compute materialization, 40 bounded planning packets/41,056 bytes, and a 75.08 ms max 1M stage. |
| `../../spatialbench/sf100/2026-08-28-native-consolidation-final/candidate.json` | `dccbff9c4e65d0ab0114cb4c68d3dacbb48bd5b9c4ccb3b53ee010df9bb238fe` | Prior-source strict-native SF100: 12/12 completed; 422.41 s total. Q6 is 12.51 s after removing the repeat-only full-zone coordinate shape; Q9 is the 0.14 s small-work exception. |
| `../../spatialbench/sf100/2026-08-28-native-consolidation-final/acceptance.json` | `ecffaf458f60c7917f7dabefba3671fd121006d5d801d1a1f6ffc907e9e25d51` | Fail-closed identity/correctness verification passes all 12 result CSVs against comparator `a75e20ba...`; 422.41 s versus 8,086.00 s, or 19.143x. Result-manifest SHA256 is `be1c690e...`. |
| `full-pipeline-exact-compaction-final.json` | `2c7fd05e1fd86fc2a2651ccdbc7c61b1c904e90ef2c00c9eeaea8de97e6412e8` | Source digest `5f2f51b0...`; 22/22 active 100K/1M lanes pass, two raster lanes are deferred, zero fallback/compute materialization, 40 bounded planning packets/41,056 bytes, and a 79.12 ms max 1M stage. |
| `../../spatialbench/sf100/2026-08-28-native-consolidation-exact-compaction-final/candidate.json` | `6faf54377fe19eefb7f8ac025f43e2a244a136695f4eb8fff142aef16b150478` | Final-source strict-native SF100: 12/12 completed; 429.00 s total. Q6 is 13.57 s after exact two-phase-admitted row-view compaction; Q9 is the 0.14 s small-work exception. |
| `../../spatialbench/sf100/2026-08-28-native-consolidation-exact-compaction-final/acceptance.json` | `608aab7e4d0fc88b418f8ea511b91f737992c80678abc14fec8f9b3061d71569` | Fail-closed identity/correctness verification passes all 12 result CSVs against comparator `a75e20ba...`; 429.00 s versus 8,086.00 s, or 18.848x. Result-manifest SHA256 is `f6bc6c7d...`. |
| `full-pipeline-final-reviewed-source.json` | `a3553cb2c3be3a32e2a77230db0c9932fb0721eb3135e61efd2567e7c4a879c2` | Source digest `1191b971...`; 22/22 active 100K/1M lanes pass, two raster lanes are deferred, zero fallback/compute materialization, 40 bounded planning packets/41,056 bytes, and a 77.58 ms max 1M stage. |
| `../../spatialbench/sf100/2026-08-28-native-consolidation-final-reviewed-source/candidate.json` | `b68ec0feef79e1deeea6cbe01130cc19a244b500ff6e7bbe80256000180f70e8` | Reviewed final-source strict-native SF100: 12/12 completed; 426.39 s total. Q6 is 13.24 s after exact two-phase-admitted row-view compaction; Q9 is the 0.14 s small-work exception. |
| `../../spatialbench/sf100/2026-08-28-native-consolidation-final-reviewed-source/acceptance.json` | `561db8c8fded181f1f531111fd20ffc5ce8f2b23b6a4db14ceedc702c10b9a14` | Fail-closed identity/correctness verification passes all 12 result CSVs against comparator `a75e20ba...`; 426.39 s versus 8,086.00 s, or 18.964x. Result-manifest SHA256 is `9540a8f2...`. |
| `full-pipeline-landing-tree.json` | `c188fddb85923d1fbcc516438979f28354cf0a80d42914bce6513ce4f18c8f15` | Source digest `2107dee8...`; 22/22 active 100K/1M lanes pass, two raster lanes are deferred, zero fallback/compute materialization, 40 bounded planning packets/41,056 bytes, and a 73.70 ms max 1M stage. |
| `../../spatialbench/sf100/2026-08-29-native-consolidation-landing-tree/candidate.json` | `4cbc9c482453cf3296088ec09ba8e9df45c8bfaa46753e2b95fc46f0290c5a0a` | Landing-tree strict-native SF100: 12/12 completed; 426.58 s total. Q6 is 13.13 s and Q9 is the 0.14 s small-work exception. |
| `../../spatialbench/sf100/2026-08-29-native-consolidation-landing-tree/acceptance.json` | `7a071e275e0d3f87e19701733198936e0bb9939de9918d84133252d877e5a4ea` | Fail-closed verification passes all 12 result CSVs against comparator `a75e20ba...`; 426.58 s versus 8,086.00 s, or 18.955x. Result-manifest SHA256 is `cc6eb16a...`. |
| `/tmp/native-consolidation-landing-strict.json` | `554d7e7286fdd3b63467e61eeb6321a41a36f0a2936b87b81756adf63200574d` | Exact landing tree: 2,239 passed, 54 classified failures, 410 skipped, 6 xfailed; 97.39% native pass rate. One admitted `build_area` lane remains open; nothing is unclassified. |
| `/tmp/cmab-flat-sindex-auto-10k-repeat3.json` | `1d8089ec07a0aa18ec3dc71c607c7af040204632bf4388ae90cad9ff918f1f00` | Exact; 0.0780 s versus 0.6963 s, 8.93x; partial device slices and flat spatial index have no geometry transfer or materialization. |
| `/tmp/power-slice-indirection-10k-repeat3.json` | `021b62a3e28eafd7ed5815d5eff477a047b25f9e2faee3f17a32c62e8e1dc061` | Exact; 0.0598 s versus 0.0425 s, 0.711x in the artifact; both `iloc` stages have zero transfer/materialization. The only runtime D2H is an 8-byte nearest allocation fence plus a 38,960-byte terminal public-index export. |
| `/tmp/native-consolidation-current/insurance-100k-current-profile.json` | `5fd57744d5f5ea00f3ca0ce6f37a459dfc0810d3552dc990564a415845a8f002` | Source digest `7cebfe3b...`; exact; 0.2353 s versus 0.1448 s, 0.621x. All 37 operations are GPU-selected with no fallback; the nominal 100K input becomes 5,601 post-clip rows and 1,131 overlay rows. No stage exceeds 98 ms. |
| `/tmp/native-consolidation-current/osm-enrichment-100k-current-profile.json` | `359c3184a56ad7da9a4587195f36eace598dc33f800d115e7806190025213c60` | Source digest `7cebfe3b...`; exact; 0.5786 s versus 0.1644 s, 0.290x. The asset is capped at 1,934 rows. All 56 operations are GPU-selected with no fallback; one 8-byte internal count fence remains. Deferred two-group multipart union materialization, not Parquet IO or CPU composition, accounts for the synchronized >1 s stage. |

Rows under `/tmp` remain session artifacts. The final full-pipeline and SF100
packets above are durable current-worktree evidence, not clean-revision approval.

## Capacity Evidence

`redevelopment-1m-geopandas-capacity.json` is durable and has SHA-256
`ae90a673d3741f8a78e04ff17ba96fc850602a1aa2af08499712da075d13ce2c`.
The isolated GeoPandas comparator completed zero samples in 27 minutes, reached
30.06 GiB cgroup memory plus 558 MiB swap, and was stopped before its 36 GiB
hard ceiling. It is capacity evidence, not a timing or correctness result. Do
not retry it in an editor/agent scope or without an explicitly approved larger
envelope.

## Post-Measurement Verification

The final session source digest, which is newer than most measurements above,
passes:

- full overlay: 398 passed, 2 skipped;
- contract health: all eight required surfaces and the optional 46-test
  performance rail pass;
- adjacent Native/shape suite: 740 passed;
- focused clip and device spatial-index suite: 259 passed;
- strict grouped sweep: 2,239 passed with only the 54 ledgered declines. The
  sweep exposed one uninitialized pickle-state field, which was fixed and then
  confirmed by the complete 126-test geodataframe file and final broad rerun;
- mandatory full pipeline: no stage above 1 second and no unexplained CPU-heavy
  stage. The final tabular-breadth audit has 22/22 active passes, zero fallback
  or compute materialization, and a 75.42 ms maximum 1M stage;
- nullable/string/categorical/temporal distinct, float null/NaN sort ordering,
  sparse grouped first/last, mixed public dissolve reducers, and bounded top-k
  pass focused strict-native cases; the broader Native/Arrow/runtime rail
  passes 808 tests. Top-k's 22-case focused matrix covers nullable fixed-width
  keys, multipart tables, primary-key skew, pandas tie order, and admission;
  the final broader Native/Arrow/runtime rail passes 811 tests.
- restricted query/eval and bounded merge/join pass the complete private native
  substrate file: 569 passed. The focused final matrix is 53 passed, including
  public validation, strict decline, admission, metadata, nullable string,
  cardinality, and real two-stream readiness canaries. Independent GPU review
  reports no blocking findings.
- variable-width indexed spatial inputs compact exactly at the consumer
  boundary after separate planning and output memory admissions. The focused
  exact-compaction rail passes 173 tests, including device-only 1,005-to-5
  coordinate sizing, decline-before-flatten, and all-null output canaries. Q6
  completes warmup plus three samples in one capped process without pool growth.
- the repository-wide pytest run completed without a process crash: 7,840
  passed, 430 skipped, and 7 xfailed. Its five Native transfer failures were
  resolved and the complete grouped-point/global-union files then pass 70/70.
  The 13 remaining broad-run failures are ten already-classified unsupported
  CRS transforms, one changed upstream Arrow xfail expectation, and two
  failures from the separately preserved `build_area` worktree addition.

The two material 100K losses are classified on the current source digest.
Insurance has only 5,601 rows after clip and 1,131 overlay rows, so its 235 ms
GPU workflow does not amortize fixed costs against a 145 ms CPU result. OSM is
asset-capped at 1,934 rows; its two output groups contain 86,426 source segments.
The synchronized profile attributes 1.445 s of deferred exact multipart-union
topology to terminal `to_parquet`, including 172,902 split events across 21
plans. That stage is GPU work realized at export, not host writer composition.

All heavy work ran one service at a time with `MemoryHigh=20G`,
`MemoryMax=28G`, and `MemorySwapMax=4G`. Three earlier hard editor/agent deaths
were diagnosed as `systemd-oomd` app-scope kills, not CUDA crashes.

## Acceptance Status

Local consolidation evidence is complete for the landing tree: the newest
runtime source passes the same-source full pipeline, M4 core/external
reconciliation, and current SF100 at 18.955x. Final hold acceptance remains
open because `build_area` is still an admitted Native gap, the core 1M
comparator is capacity-limited, and the clean revision, CI references, and
explicit maintainer approval are not yet recorded.
