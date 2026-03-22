# Benchmarking

vibeSpatial ships a unified benchmarking CLI called **`vsbench`**.  Use it to
measure individual operations, run regression suites, compare against
GeoPandas, and generate HTML reports.

## Quick start

```bash
# List available operations
vsbench list operations

# Run a single operation benchmark
vsbench run buffer --scale 100k

# Run the smoke suite (fast, good for local dev)
vsbench suite smoke

# Compare vibeSpatial vs GeoPandas on your own script
vsbench shootout my_workflow.py
```

## Commands

### `vsbench run <operation>`

Run a single operation benchmark at a given scale.

```bash
vsbench run intersects --scale 10k --repeat 5
vsbench run buffer --scale 1m --precision fp32 --gpu-sparkline
vsbench run dissolve --compare geopandas --json
```

**Common flags** (shared by `run`, `pipeline`, `suite`, `kernel`):

| Flag | Default | Description |
|------|---------|-------------|
| `--scale` | operation default | Input size: `1k`, `10k`, `100k`, `1m` |
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
vsbench list operations                # All registered operations
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
vsbench shootout examples/nearby_buildings.py --repeat 5
vsbench shootout my_etl.py --with pyogrio --json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--repeat` | 3 | Timed runs per engine |
| `--no-warmup` | off | Skip the untimed warmup run |
| `--baseline-python` | auto | Python interpreter with real geopandas |
| `--with DEP` | none | Extra pip deps for the geopandas env (repeatable) |
| `--timeout` | 300 | Per-run timeout in seconds |

## Fixture scales

| Name | Rows |
|------|------|
| `1k` | 1,000 |
| `10k` | 10,000 |
| `100k` | 100,000 |
| `1m` | 1,000,000 |

Fixtures are cached in `.benchmark_fixtures/` and auto-generated on first
use. Use `vsbench fixtures generate` to pre-populate.
