# Make Valid Pipeline

<!-- DOC_HEADER:START
Scope: Compact-invalid-row make_valid pipeline staging and repair-only-invalids policy.
Read If: You are changing make_valid, validity checking, or topology repair pipelines.
STOP IF: Your task already has the make_valid pipeline open and only needs local implementation detail.
Source Of Truth: Make-valid pipeline architecture for compact-and-repair staging.
Body Budget: 110/220 lines
Document: docs/architecture/make-valid.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-17 | Request Signals |
| 18-24 | Open First |
| 25-30 | Verify |
| 31-36 | Risks |
| 37-65 | Decision |
| 66-88 | Dispatch |
| 89-110 | Performance Notes |
DOC_HEADER:END -->

## Intent

Define the repo-owned `make_valid` pipeline so topology repair runs only on
logically selected invalid device rows and returns an atomic native result.

## Request Signals

- make_valid
- validity
- topology repair
- compaction
- invalid rows
- shapely.make_valid
- geometryarray make_valid

## Open First

- docs/architecture/make-valid.md
- src/vibespatial/constructive/make_valid_pipeline.py
- src/vibespatial/api/_shapely_dispatch.py
- tests/test_make_valid_pipeline.py

## Verify

- `uv run pytest tests/test_make_valid_pipeline.py`
- `uv run pytest tests/test_device_geometry_array.py -k shapely_make_valid_dispatches_device_geometry_array_directly`
- `uv run python scripts/check_docs.py --check`

## Risks

- Running repair on inactive capacity lanes wastes compute on already-valid geometries.
- Validity checking and repair becoming coupled prevents staging them as separate GPU stages.
- Undocumented third-party adapter hooks make import-time behavior hard to discover.

## Decision

- Compute validity first.
- Represent invalid rows with `NativeDeviceSelection` at source capacity.
- Leave valid rows untouched.
- Repair only active selection lanes while retaining bounded row/ring capacity.
- Scatter repaired rows back through one row-indirected capacity carrier;
  inactive lanes route to scratch destinations instead of sizing a compact copy.
- Keep validity, invalid-family, family/global, and repaired mappings as aligned
  device rowsets. Native repair returns a complete aligned carrier or declines
  atomically; callers never patch residual rows with host geometry.
- Repair invalid MultiPolygon rows by exploding polygon parts, repairing parts,
  reducing them with grouped fp64 constructive topology, and scattering the
  grouped rows back through the logical row mapping.
- In `linework` mode, preserve lower-dimensional collapsed/internal boundary
  output with `NativeGeometryComposition`: repaired polygonal area and source
  boundary minus repaired-area boundary remain concrete native parts until
  terminal public GeometryCollection assembly.
- When constructing a replacement ``GeoSeries`` from repaired geometry,
  always pass ``index=df.index`` (or ``index=gs.index``) to preserve
  non-contiguous index alignment from upstream operations like ``clip()``
  or ``iloc`` slicing.  Omitting the index creates a default
  ``RangeIndex(0..N-1)`` which silently drops rows during pandas column
  assignment when the DataFrame index is non-contiguous.
- When all rows pass validation and an ``OwnedGeometryArray`` was provided,
  ``MakeValidResult.owned`` carries the original device-resident array so
  downstream stages (e.g., dissolve) can stay on device without re-uploading
  (ADR-0005 zero-transfer chain).

## Dispatch

- ``make_valid_owned()`` owns runtime dispatch via ``plan_dispatch_selection()``
  and records dispatch events internally; the API layer (``GeometryArray``,
  ``DeviceGeometryArray``) does not record its own events.
- Two kernel variants are registered (ADR-0033):
  ``make_valid/gpu-nvrtc`` (polygon/multipolygon GPU repair) and
  ``make_valid/cpu`` (all families, Shapely fallback).
- The ``dispatch_mode`` parameter controls GPU/CPU/AUTO selection.
- A native repair decline records the whole-operation CPU boundary before any
  host materialization. Strict-native mode rejects the crossover without
  leaking a partial device result or performing the transfer first.
- vibeSpatial installs a process-wide ``shapely.make_valid`` adapter at import
  time via ``src/vibespatial/api/_shapely_dispatch.py``. For repo-owned
  ``GeometryArray`` and ``DeviceGeometryArray`` inputs, the wrapper dispatches
  to ``geometry.make_valid(...)`` so device-backed public workflows such as
  ``gdf.set_geometry(shapely.make_valid(gdf.geometry.values))`` stay on the
  native path. All other input types continue to use Shapely's original
  implementation.
- This hook is limited to ``make_valid``. New Shapely monkeypatches should not
  be added casually; if a public adapter hook is necessary, document it here or
  in a dedicated ADR before landing.

## Performance Notes

- Validity checking is much cheaper than topology repair, so device selections
  exclude valid rows from topology work without physically compacting geometry.
- Device validity expressions become capacity selections; duplicate indexed
  logical rows stay aligned through repair and row-indirected scatter.
- GPU repair establishes device state once at entry. The obsolete host
  coordinate/offset batch builder and Python ring/geometry reconstruction path
  are deleted; normalized repair rows have one nested-buffer contract.
- Polygon and multipart area repair stays device-resident. Unsupported native
  repair shapes decline as a whole; host geometry is otherwise materialized
  only at an explicit whole-operation compatibility or terminal export boundary.
- Ring closure allocates the structural upper bound of one extra coordinate per
  ring. Duplicate removal scans and scatters into retained coordinate capacity;
  ring offsets carry the logical active prefix into later native consumers.
- Repaired-ring filtering keeps ring and coordinate capacity, weights grouped
  ring counts by the active mask, and passes the device logical count into the
  gathered-buffer carrier. Repair completion has one explicit atomic admission
  scalar; there are no per-family compact-length reads.
- Invalid normalized rows polygonize through overlay's paged segment sweep,
  streamed split-event merge, half-edge graph, and positive bounded-face
  selector. Make-valid no longer has an active quadratic split/rebuild engine.
