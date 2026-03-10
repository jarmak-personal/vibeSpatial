<!-- DOC_HEADER:START -->
> [!IMPORTANT]
> This block is auto-generated. Edit metadata in `docs/doc_headers.json`.
> Refresh with `uv run python scripts/check_docs.py --refresh` and validate with `uv run python scripts/check_docs.py --check`.

**Scope:** IO format classification as GPU-native, hybrid, or fallback-only.
**Read If:** You are classifying a format pathway or adding a new IO format target.
**STOP IF:** Your task already has the specific format adapter open and only needs local implementation detail.
**Source Of Truth:** IO format classification matrix for GPU-native versus hybrid versus fallback paths.
**Body Budget:** 53/220 lines
**Document:** `docs/architecture/io-support-matrix.md`

**Section Map (Body Lines)**
| Body Lines | Section |
|---|---|
| 1-5 | Intent |
| 6-13 | Request Signals |
| 14-19 | Open First |
| 20-23 | Verify |
| 24-28 | Risks |
| 29-36 | Decision |
| 37-47 | Matrix |
| 48-53 | Performance Notes |
<!-- DOC_HEADER:END -->

## Intent

Define which formats are GPU-native targets, which are acceptable hybrid paths,
and which remain explicit fallback adapters.

## Request Signals

- io support matrix
- format classification
- gpu native format
- hybrid path
- fallback format

## Open First

- docs/architecture/io-support-matrix.md
- docs/architecture/io-arrow.md
- docs/architecture/io-files.md

## Verify

- `uv run python scripts/check_docs.py --check`

## Risks

- Compatibility formats becoming the design center instead of GeoArrow and GeoParquet.
- Missing explicit fallback events on legacy GDAL formats hides work that bypasses the GPU stack.

## Decision

- GeoArrow is the canonical GPU-native interchange format.
- GeoParquet is the canonical GPU-native persisted format.
- WKB is a hybrid compatibility bridge.
- GeoJSON and Shapefile are explicit hybrid pipelines.
- Untargeted GDAL formats remain fallback-only until justified.

## Matrix

| Format | Default Path | Read | Write | GPU Canonical |
|---|---|---|---|---|
| GeoArrow | GPU-native | GPU-native | GPU-native | yes |
| GeoParquet | GPU-native | GPU-native | hybrid | yes |
| WKB | hybrid | hybrid | hybrid | no |
| GeoJSON | hybrid | GPU geometry + CPU properties | hybrid | no (geometry GPU-accelerated) |
| Shapefile | hybrid | hybrid | hybrid | no |
| GDAL legacy | fallback | fallback | fallback | no |

## Performance Notes

- Most serious IO throughput work should center on GeoArrow and GeoParquet.
- Compatibility formats must not become the design center.
- Hybrid paths are acceptable when transport or container parsing is host-heavy
  but geometry decode, filtering, or encode can still move toward GPU stages.
