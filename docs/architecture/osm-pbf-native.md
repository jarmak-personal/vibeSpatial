# Native OSM PBF Ingress

<!-- DOC_HEADER:START
Scope: GPU PBF byte decoding, relation topology, native carriers, and bounded-memory admission.
Read If: You are changing OSM PBF performance, layer semantics, decompression, or relation memory use.
STOP IF: You only need a public read_file argument documented in io-files.md.
Source Of Truth: Native PBF implementation contract and measurement boundaries.
Body Budget: 184/220 lines
Document: docs/architecture/osm-pbf-native.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-16 | Request Signals |
| 17-29 | Open First |
| 30-35 | Verify |
| 36-42 | Risks |
| 43-77 | Public Contract |
| 78-119 | Physical Workload Shape |
| 120-154 | Memory Shape |
| 155-178 | Comparison Boundary |
| 179-184 | References |
DOC_HEADER:END -->

## Intent

Decode standard Geofabrik PBF files directly into native GPU geometry and
attribute columns, including polygon relations and ordered collections.

## Request Signals

- osm pbf
- geofabrik
- nvcomp deflate
- relation ring assembly
- pbf memory
- pbf benchmark

## Open First

- src/vibespatial/io/osm_pbf_native.py
- src/vibespatial/io/osm_pbf_inflate.py
- src/vibespatial/io/osm_pbf_relations.py
- src/vibespatial/io/osm_pbf_rings.py
- src/vibespatial/io/osm_pbf_memory.py
- scripts/benchmark_osm_pbf.py
- scripts/benchmark_osm_pbf_capacity.py
- docs/testing/osm-pbf-performance.md
- tests/test_osm_pbf_native.py
- tests/test_osm_pbf_relations.py

## Verify

- `uv run pytest tests/test_osm_pbf_native.py tests/test_osm_pbf_relations.py tests/test_io_file.py --run-gpu -q`
- `uv run python scripts/benchmark_osm_pbf.py DATA.osm.pbf --cache /tmp/osm-comparator --output /tmp/osm-current.json --repeat 3`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- File bytes, node counts, and expanded relation coordinates are different memory budgets.
- GeometryCollection public export still materializes Shapely; native ingress does not.
- CUDA managed allocations use host backing and are outside the ordinary RMM pool counter.
- OSM extracts can contain incomplete relations and invalid polygon rings.

## Public Contract

Eligible local, unfiltered `read_file` requests use nvCOMP 5 Deflate and NVRTC
protobuf codecs with device-native pylibcudf strings. The RAPIDS dependency
groups supply pylibcudf and nvCOMP. Raw Blobs also work without decompression.
All five standard GDAL OSM layers are admitted:

| Layer | Geometry |
|---|---|
| points | Significant tagged Point nodes |
| lines | Tagged non-area LineString ways |
| multilinestrings | Route and multilinestring relations |
| multipolygons | Polygon/boundary relations followed by standalone area ways |
| other_relations | Ordered GeometryCollections of node and way members |

The default and `layer="all"` concatenate these layers in that order and add
`osm_element` (`node`, `way`, `relation`). `tags=False` retains identifiers;
`geometry_only=True` omits all attributes. An OSM standard layer does not
produce MultiPoint, although collections may contain many point members.
Coordinates are EPSG:4326; `target_crs` performs actual GPU reprojection.

The compatibility contract is GDAL 3.11.4's default `osmconf.ini`: promoted
attributes, HSTORE escaping, significant tags, area classification, outer-way
suppression, and inherited outer-way polygon tags. Nested relation members
are ignored, matching that driver. Missing polygon members suppress the
relation; route and collection reads retain resolved members. The legitimate
coordinate (0,0) is preserved despite GDAL's node-index sentinel bug.

Explicit engines, filters, custom GDAL OSM configuration, unsupported required
features/compression, ordinary non-dense Nodes, split packed fields, and
unordered node blocks or standalone polygon way IDs use the observable compatibility admission boundary.
Malformed framing, invalid byte spans, zlib checksums, and source replacement
are checked. A synchronous completion fence makes results safe for consumers
on another CUDA stream.

## Physical Workload Shape

This is a private codec behind `read_vector_file_native`, not a row-wise
geometry dispatch kernel. Its reusable shapes are byte/block traversal,
segmented delta decode, relation consumption, and dynamic output assembly.

| Stage | Work units and layout | Primitive |
|---|---|---|
| Inflate | Compressed/raw Blob bytes; bounded byte arenas | nvCOMP batched Deflate |
| Parse | Blob/string/record spans; separate control lane per Blob | Tier 1 NVRTC |
| Delta decode | Packed bytes and node coordinates; warp prefixes | Tier 1 NVRTC |
| Attribute emit | Output UTF-8 bytes, offsets and validity masks | Tier 1 codec + pylibcudf carriers |
| Count/scan | Integer counts and CSR offsets | Tier 3 CCCL AUTO, asynchronous |
| Resolve | Output references, sorted node-block ranges, sharded links | Tier 1 indexed gather |
| Assemble | Relation members, endpoint states, rings, parent candidates | Tier 1 graph/topology + Tier 2 array transforms |
| Finalize | Attribute columns and geometry compositions | pylibcudf concat + native buffer adoption |

Scans request existing CCCL warmup; AUTO uses the measured CuPy path while
CCCL is cold. Composite endpoint/ring ordering uses CuPy lexsort because the
current CCCL wrapper accepts scalar keys, and a multi-pass scalar-key sort
would add staging and launches. Integer identifiers stay int64. Offsets are
counted in int64 and checked before the owned int32 representation.

Coordinates use the driver's 100-nanodegree lattice in fp64. This byte codec
cannot downcast authoritative coordinates. Ring construction selects an
observable CONSTRUCTIVE fp64 PrecisionPlan; boundary/midpoint tests explicitly
round binary64 operations to preserve OGR topology decisions. Area ordering
uses integer lattice arithmetic and a 128-bit accumulator. Kernels register
in NVRTC warmup under `osm-pbf-native-fp64`.

Inputs are a mapped file and bounded typed byte/metadata arrays. Outputs are
`NativeTabularResult`, pylibcudf-backed `NativeAttributeTable`, owned family
buffers, and `NativeGeometryComposition`. Collection parts carry a device
vector of member positions, so Python composition parts scale with geometry
families instead of the largest collection's member count. Open endpoints use
one sorted group directory. Shared junctions advance monotone per-group
cursors, so greedy traversal visits each endpoint at most once; it does not
scan the full junction again for every member. Position vectors
survive row selection, duplication, masking, concatenation and CRS changes.
Host transfers inside ingress contain allocation sizes, validation status,
and block-selection metadata; coordinates, member IDs and tags stay on GPU.

## Memory Shape

Blob batches target at most 256 MiB of decompressed bytes, further limited by
a conservative decode expansion estimate and freshly queried physical VRAM.
The admission takes the minimum of driver free bytes and the allocator budget;
a previously captured pool envelope cannot hide another application's VRAM use.
A minimum indivisible Blob can use managed workspace under pressure.

Point reads retain chunks only while measured free memory permits the final
concatenation. Otherwise they discard tentative output and count/replay into
one final allocation. Ways count selected references, allocate final x/y,
then replay selected way Blobs. During reference resolution x holds node IDs
and y holds intrusive links; both are overwritten with coordinates. Only
referenced node Blobs are decoded. There is no complete node-ID/coordinate
index: directory storage scales with DenseNodes blocks and is sharded to an
8 MiB ordinary-case head budget.

Relations first match requested ways using per-Blob ID bounds. Windows are
sized by expanded coordinates and member graph bytes, capped at 256 MiB,
rather than compressed input size. A single oversized relation uses explicit
CUDA-managed workspace. Final geometry concatenation adopts existing buffers;
attribute concatenation can use managed destination columns when a second
resident copy would exceed available device memory. Dispatch events identify
these paging decisions. Output arrays can also use managed allocation when
normal final storage would overlap too much live scratch. Inherited polygon
tags admit actual expanded string bytes: repeated outer-way tags plus both
libcudf gather and destination scatter copies. GPU computation remains on the device, while the CUDA
driver can page managed storage through host RAM.

These mechanisms bound ordinary intermediates and cover skewed relations.
They require enough host backing for managed pages and enough memory for the
final native representation; transient working sets can page through host RAM. Owned offsets and
libcudf string-column size limits remain explicit capacity boundaries. RMM
peak statistics alone do not measure managed pages or CUDA context overhead.

## Comparison Boundary

The benchmark records a SHA256-identified PBF, package/GPU/CPU identity,
immutable GDAL comparators, preconverted native GeoArrow GeoParquet, cold
calls, warm medians, RMM peaks, and process RSS. Conversion and full
attribute/geometry correctness fingerprints are outside timed sections.
Candidate reruns reuse matching comparator artifacts. `--pool-limit BYTES`
constrains only candidate subprocesses. `benchmark_osm_pbf_capacity.py pressure`
reserves physical VRAM outside RMM before its pool grows; its `junction` mode
checks 10K/100K/1M members sharing one endpoint. See the
[measured report](../testing/osm-pbf-performance.md) for results and caveats.

Point, line and multipart GeoParquet comparisons use native GeoArrow encoding,
Snappy, and one-million-row groups. GeoParquet 1.1 has no native
GeometryCollection encoding, so the collection layer reports its GDAL
comparison without a misleading native-GeoArrow timing. Polygon fingerprints
normalize ring starts/orientation and retain invalid geometry, for which
GEOS topological equality may reject even identical inputs.

Native-payload and public-GeoDataFrame timings are separate. Standard family
exports can retain device geometry; collection public export uses the existing
terminal Shapely compatibility boundary. A default all-layer public export
therefore includes that materialization cost.

## References

- [OSM binary schema](https://github.com/openstreetmap/OSM-binary/tree/master/osmpbf)
- [GDAL 3.11.4 OSM implementation](https://github.com/OSGeo/gdal/blob/v3.11.4/ogr/ogrsf_frmts/osm/ogrosmdatasource.cpp)
- [CCCL Python primitives](https://nvidia.github.io/cccl/unstable/python/compute_api.html)
- [GeoParquet 1.1 encodings](https://geoparquet.org/releases/v1.1.0/)
