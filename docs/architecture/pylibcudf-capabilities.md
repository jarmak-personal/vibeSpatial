# pylibcudf Capabilities

<!-- DOC_HEADER:START
Scope: Confirmed pylibcudf/libcudf/KvikIO Parquet source and scan capabilities used by vibeSpatial IO.
Read If: You are changing GeoParquet transport policy, SourceInfo wiring, or backend capability gates.
STOP IF: The routed GeoParquet adapter files are already open and you only need a local implementation detail.
Source Of Truth: Checked-in capability matrix for what the current pylibcudf Parquet reader actually supports in this repo.
Body Budget: 0/260 lines
Document: docs/architecture/pylibcudf-capabilities.md
DOC_HEADER:END -->

## Intent

Separate confirmed backend capability from repo policy. This document tracks
what the current `pylibcudf` Parquet reader can actually do when exercised from
vibeSpatial instead of inferring support from the Python surface alone.

## Request Signals

- pylibcudf capability matrix
- kvikio integration
- SourceInfo support
- GeoParquet backend gate
- remote parquet support

## Open First

- docs/architecture/pylibcudf-capabilities.md
- docs/architecture/io-arrow.md
- src/vibespatial/io/geoparquet.py
- tests/test_pylibcudf_capabilities.py

## Verify

- `uv run pytest tests/test_pylibcudf_capabilities.py -q`
- `uv run pytest tests/test_io_arrow.py -q -k "file_uri or partitioned_directory_uses_gpu_scan_backend"`
- `uv run python scripts/check_docs.py --check`

## Risks

- Treating the Cython API as proof of support can overstate what the backend
  really handles in practice.
- Conflating repo policy with backend gaps makes it harder to see when the next
  step is a wiring change instead of a new scanner.
- URI support is not one bucket: normalized local `file://` behavior, raw
  `SourceInfo` URI handling, and authenticated remote transport have different
  constraints.

## Confirmed Parquet Read Capabilities

These are exercised by repo-local probes in
[`tests/test_pylibcudf_capabilities.py`](/home/picard/repos/vibeSpatial/tests/test_pylibcudf_capabilities.py).

| Capability | Status | Notes |
|---|---|---|
| Local path source | confirmed | `SourceInfo([path])` works directly. |
| Raw `bytes` source | confirmed | `SourceInfo([bytes])` works directly. |
| `BytesIO` source | confirmed | `SourceInfo([BytesIO])` works directly. |
| `rmm.DeviceBuffer` source | confirmed | `SourceInfo([DeviceBuffer])` works directly. |
| Column projection | confirmed | `ParquetReaderOptions.set_columns(...)` works. |
| Row-group selection | confirmed | `set_row_groups(...)` works for single-source and multi-source reads. |
| Predicate filters | confirmed | `set_filter(...)` works with compiled expressions. |
| Multi-source scan | confirmed | Multi-file `SourceInfo([...])` plus per-source row groups works. |
| `ChunkedParquetReader` | confirmed | Works off `ParquetReaderOptions` and preserves filtering. |

## Confirmed URI And Transport Behavior

| Capability | Status | Notes |
|---|---|---|
| Public `file://` GeoParquet read | confirmed | vibeSpatial normalizes the URI to a local path before handing it to `pylibcudf`; the public read stays on the GPU scan backend. |
| Local partitioned directory GeoParquet read | confirmed | Public `read_parquet()` stays on the `pylibcudf` scan backend for local partitioned datasets. |
| Raw `SourceInfo([file://...])` | unsupported | libcudf/KvikIO treats it as a remote endpoint and rejects it. |
| Raw `SourceInfo([http://...])` | conditional / unresolved | The API surface exists, but a simple local HTTP server probe failed because the endpoint did not satisfy KvikIO remote-range expectations. |

## Repo Policy Versus Backend Gaps

- Confirmed repo-supported GPU scan cases today:
  - local files
  - normalized `file://` public reads
  - local partitioned directories
  - in-memory `bytes` and `BytesIO`
- Still explicit repo policy fallback today:
  - custom filesystems
  - `storage_options`
  - authenticated remote transports
- Backend surface exists but is not yet a repo contract:
  - raw remote URIs through `SourceInfo`
  - `Datasource`-backed reads

The current repo stance is deliberate: do not advertise a GPU transport path
until metadata planning, auth, error handling, and fallback observability are
defined at the public read boundary.
