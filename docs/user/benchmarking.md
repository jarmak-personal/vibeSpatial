# Benchmarking

vibeSpatial ships a unified benchmarking CLI called **`vsbench`**.  Use it to
measure individual operations, run regression suites, compare against
GeoPandas, and generate HTML reports. For repo-local GPU and strict-native
workflows, invoke it through `uv run vsbench ...` so the project-managed
environment and GPU runtime detection stay intact.

## Quick start

```bash
# List available operations
uv run vsbench list operations

# Run a single operation benchmark
uv run vsbench run buffer --scale 100k

# Run the smoke suite (fast, good for local dev)
uv run vsbench suite smoke

# Compare vibeSpatial vs GeoPandas on your own script
uv run vsbench shootout my_workflow.py
```

## Commands

### `vsbench run <operation>`

Run a single operation benchmark at a given scale.

```bash
vsbench run intersects --scale 10k --repeat 5
vsbench run buffer --scale 1m --precision fp32 --gpu-sparkline
vsbench run dissolve --compare geopandas --json
vsbench run bounds-pairs --rows 20000 --arg dataset=uniform --arg tile_size=256
vsbench run clip-rect --arg kind=polygon --arg rect=100,100,700,700
```

Use `vsbench list operations --json` to discover typed operation-specific
parameters. Operation arguments are passed with repeatable `--arg key=value`
flags and validated against the operation schema before execution.
Default operation listings and suites measure public GeoPandas-compatible APIs;
private owned-array and kernel diagnostics require `--include-internal` or
`vsbench kernel`.

**Common flags** (shared by `run`, `pipeline`, `suite`, `kernel`):

| Flag | Default | Description |
|------|---------|-------------|
| `--scale` | operation default | Input size: `1k`, `10k`, `100k`, `1m` |
| `--rows` | none | Exact input row count override for `vsbench run` |
| `--repeat` | 3 | Number of timed iterations |
| `--precision` | `auto` | Force `fp32`, `fp64`, or `auto` |
| `--compare` | none | Compare against `shapely` or `geopandas` |
| `--input-format` | `parquet` | Fixture format: `parquet`, `geojson`, `shapefile`, `gpkg` |
| `--gpu-sparkline` | off | Show per-stage GPU utilization sparkline |
| `--nvtx` | off | Emit NVTX ranges for Nsight profiling |
| `--trace` | off | Enable execution trace warnings |
| `--json` | off | Output results as JSON |
| `--quiet` | off | Minimal output |
| `--output PATH` | none | Write results to file |

### `vsbench pipeline <name>`

Run a named multi-stage pipeline benchmark (e.g. spatial-join, overlay,
nearby-buildings).

```bash
vsbench pipeline spatial-join --suite-level ci
vsbench pipeline overlay --gpu-sparkline
```

The `--suite-level` flag (`smoke`, `ci`, `full`) controls which scale
points the pipeline runs at.

### `vsbench suite {smoke,ci,full}`

Run a predefined suite of benchmarks.

```bash
vsbench suite smoke                       # Fast check (~30s)
vsbench suite ci --json --output ci.json  # CI gate
vsbench suite full --gpu-sparkline        # Full profiling run
```

Suites run serially and isolate each benchmark item in a subprocess by default
so CUDA allocator state, failed kernels, or OOMs do not contaminate later
items. Use `--in-process` only for local debugging. Use `--item-timeout N` to
override the default 600 second per-item timeout.

Use `--pipeline <name>` (repeatable) to limit to specific pipelines.

### `vsbench kernel <name>`

Run a Tier-2 NVBench kernel microbenchmark (requires `cuda-bench`).

```bash
vsbench kernel point_in_polygon --scale 100k --bandwidth
```

The `--bandwidth` flag reports GB/s and percent-of-peak memory bandwidth.

### `vsbench compare <baseline> <current>`

Detect performance regressions between two JSON result files.

```bash
vsbench suite ci --json --output baseline.json
# ... make changes ...
vsbench suite ci --json --output current.json

vsbench compare baseline.json current.json
```

Returns exit code 1 if regressions are detected.

### `vsbench report <results>`

Generate an HTML report from a JSON result file.

```bash
vsbench report ci.json -o report.html
```

### `vsbench list {operations,pipelines,fixtures,kernels}`

Discover what benchmarks are available.

```bash
vsbench list operations                # Public API operations
vsbench list operations --json         # Includes operation parameter schemas
vsbench list operations --include-internal  # Include private diagnostics
vsbench list operations --category io  # Filter by category
vsbench list pipelines                 # Available pipelines
vsbench list fixtures                  # Fixture specs and scales
vsbench list kernels                   # NVBench kernel benches (Tier 2)
```

### `vsbench fixtures generate`

Pre-generate benchmark fixture files (synthetic datasets in various
formats and scales).

```bash
vsbench fixtures generate                           # All scales, all formats
vsbench fixtures generate --scale 100k --format parquet
vsbench fixtures generate --force                   # Regenerate even if cached
```

### `vsbench shootout <script.py>`

Head-to-head comparison of a Python script running under real GeoPandas vs
vibeSpatial.  The script should `import geopandas as gpd` -- vsbench
handles the import swap transparently.

```bash
uv run vsbench shootout benchmarks/shootout/nearby_buildings.py --scale 10k --repeat 3
uv run vsbench shootout benchmarks/shootout/accessibility_redevelopment.py --scale 10k --repeat 3
uv run vsbench shootout examples/nearby_buildings.py --repeat 5
uv run vsbench shootout my_etl.py --with pyogrio --json
```

The `script` argument accepts a single Python file or a directory of scripts.
For GPU and `VIBESPATIAL_STRICT_NATIVE=1` runs, `uv run vsbench shootout ...`
is the supported launch mode.

| Flag | Default | Description |
|------|---------|-------------|
| `--repeat` | 3 | Timed runs per engine |
| `--scale` | none | Passed as `VSBENCH_SCALE` env var (e.g. `10K`, `100K`) |
| `--no-warmup` | off | Skip the untimed script warmup run; the vibespatial leg still prewarms registered GPU pipelines before timing |
| `--baseline-python` | auto | Python interpreter with real geopandas |
| `--with DEP` | none | Extra pip deps for the geopandas env (repeatable) |
| `--timeout` | 300 | Per-run timeout in seconds |

For workflow parity checks, treat `--no-warmup --repeat 1` as a cold-start
probe, not a steady-state benchmark. The default `--repeat 3` is the right
floor for judging parity on the top-level workflow shootouts.

#### Fingerprint correctness checking

Scripts can print a deterministic summary line to stdout:

```
SHOOTOUT_FINGERPRINT: rows=998 bounds=(-9.55, 71.1, 1004.26, 1010.0) convex_hull_area=105251.17
```

When both engines emit a fingerprint, vsbench compares them with numeric
tolerance (`rtol=1e-3`) to catch correctness regressions while allowing
expected floating-point divergence between GPU and CPU paths.  A mismatch
is reported as a test failure.

## Fixture scales

| Name | Rows |
|------|------|
| `1k` | 1,000 |
| `10k` | 10,000 |
| `100k` | 100,000 |
| `1m` | 1,000,000 |

Fixtures are cached in `.benchmark_fixtures/` and auto-generated on first
use. Use `vsbench fixtures generate` to pre-populate.
