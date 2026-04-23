# Gridding module

The {py:mod}`AEIC.gridding` module converts simulated trajectory data into
three-dimensional gridded emissions inventories. This is the final stage of the
AEIC pipeline: trajectory stores produced by `aeic run` are binned onto a
latitude/longitude/altitude grid and written to a CF-compliant NetCDF4 file
with one variable per chemical species.

Gridding uses a **map-reduce** architecture so that it can be parallelized
across many machines. In the *map* phase, each parallel worker processes a
slice of the trajectories and writes an intermediate zarr file. In the *reduce*
phase, a single process sums all the zarr slices and writes the final NetCDF
output.

## The `trajectories-to-grid` command

The `aeic trajectories-to-grid` CLI command drives the gridding pipeline. It is
invoked with either `--mode map` or `--mode reduce`.

### Command-line options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--input-store` | Path | Yes | | Input trajectory store (`.aeic-store` directory) |
| `--grid-file` | Path | Yes | | Grid definition file (TOML) |
| `--mode` | `map` or `reduce` | Yes | | Processing mode |
| `--map-prefix` | String | Yes | | Path prefix for intermediate zarr files |
| `--mission-db-file` | Path | No | | Mission database file (SQLite). Required when using `--filter-file` or in reduce mode |
| `--filter-file` | Path | No | | Trajectory filter definition file (TOML) |
| `--output-file` | Path | No | | Final NetCDF output path. Required in reduce mode |
| `--output-times` | `annual`, `monthly`, or `daily` | No | `annual` | Output time resolution. Only `annual` is currently implemented |
| `--slice-count` | Integer | No | `1` | Number of parallel processing slices (map phase only) |
| `--slice-index` | Integer | No | `0` | Zero-based index of the slice to process (map phase only) |

### Map mode

Map mode (`--mode map`) processes a subset of trajectories from the input store
and accumulates their emissions onto the grid. The output is a single zarr
file named `{map-prefix}-{slice-index:05d}.zarr`.

A minimal single-process invocation:

```shell
aeic trajectories-to-grid \
  --input-store output/trajectories.aeic-store \
  --grid-file src/AEIC/data/grids/era5.toml \
  --mode map \
  --map-prefix output/grid/map-slice
```

This produces `output/grid/map-slice-00000.zarr`.

### Reduce mode

Reduce mode (`--mode reduce`) discovers all zarr slice files matching the
`{map-prefix}-NNNNN.zarr` pattern, validates that they were all produced with
the same grid and filter settings, sums them, and writes the final NetCDF
output.

```shell
aeic trajectories-to-grid \
  --input-store output/trajectories.aeic-store \
  --grid-file src/AEIC/data/grids/era5.toml \
  --mission-db-file oag-2019.sqlite \
  --mode reduce \
  --map-prefix output/grid/map-slice \
  --output-file output/inventory.nc
```

The `--mission-db-file` is required in reduce mode because the mission database
is queried to determine the inventory time range (earliest and latest scheduled
departure timestamps).

## Grid definition files

Grid definitions are TOML files with three sections: `[latitude]`,
`[longitude]`, and `[altitude]`. Several built-in grid definitions are shipped
in `src/AEIC/data/grids/`.

### Horizontal axes

The `[latitude]` and `[longitude]` sections share the same structure:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `resolution` | float | | Cell width in degrees |
| `range` | `[min, max]` | `[-90, 90]` (lat) or `[-180, 180]` (lon) | Domain extent in degrees |

Cell edges are placed at `range[0], range[0] + resolution, ...` and cell
centers fall at the midpoints (i.e. at `range[0] + 0.5 * resolution`,
`range[0] + 1.5 * resolution`, etc.).

### Vertical axis

The `[altitude]` section is discriminated by its `mode` field.

**Height mode** (`mode = "height"`) defines a uniform vertical grid in meters:

| Key | Type | Description |
|-----|------|-------------|
| `mode` | `"height"` | Selects uniform height grid |
| `resolution` | float | Cell height in meters |
| `range` | `[min, max]` | Altitude extent in meters |

**ISA pressure mode** (`mode = "isa_pressure"`) defines a non-uniform vertical
grid using explicit pressure levels:

| Key | Type | Description |
|-----|------|-------------|
| `mode` | `"isa_pressure"` | Selects ISA pressure level grid |
| `levels` | list of floats | Pressure levels in hPa |

When using `isa_pressure` mode, trajectory altitudes (in meters) are converted
to ISA pressure values before binning. The output NetCDF stores pressure levels
in descending order (following the ERA5 convention).

### Examples

A 1-degree uniform height grid (`basic-1x1.toml`):

```toml
[longitude]
resolution = 1.0

[latitude]
resolution = 1.0

[altitude]
mode = "height"
resolution = 500
range = [0, 15500]
```

An ERA5-compatible pressure level grid (`era5.toml`):

```toml
[latitude]
resolution = 1.0

[longitude]
resolution = 1.0

[altitude]
mode = "isa_pressure"
levels = [500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100]
```

## Filtering trajectories

By default, all trajectories in the input store are gridded. To process only a
subset, provide a `--filter-file` (TOML) along with `--mission-db-file`. The
filter file defines conditions that are combined with AND logic to select
matching flights from the mission database.

```shell
aeic trajectories-to-grid \
  --input-store output/trajectories.aeic-store \
  --mission-db-file oag-2019.sqlite \
  --filter-file my-filter.toml \
  --grid-file src/AEIC/data/grids/era5.toml \
  --mode map \
  --map-prefix output/grid/map-slice
```

Filter files use the same {py:class}`Filter <AEIC.missions.Filter>` model
described in the missions documentation. Available filter fields include
distance ranges, seat capacity, airport codes, countries, continents, and
geographic bounding boxes.

```{eval-rst}
.. note::
   The ``--filter-file`` option is only supported in map mode. In reduce mode,
   the filter metadata embedded in each zarr slice is used for consistency
   validation instead.
```

## Map-reduce architecture

### Map phase

The map phase iterates over trajectories and accumulates per-segment emissions
onto the grid:

1. **Load trajectories** from the store, either sequentially
   (`iter_range`) or by flight ID (`iter_flight_ids` when a filter is active).
2. **Split at the antimeridian** using `Trajectory.dateline_split()`.
   Trajectories that cross the +/-180 degree longitude line are split into
   sub-trajectories at the crossing point, with emissions distributed
   proportionally.
3. **Extract segments.** Each sub-trajectory of N points yields N-1 segments.
   For ISA pressure grids, segment altitudes are converted from meters to hPa.
4. **Batch and dispatch.** Segments are accumulated into batches of 1000 before
   being passed to the Numba-jitted voxel traversal kernel. The kernel uses a
   fast path for segments that stay within a single grid cell (the common case)
   and an Amanatides-Woo 3D traversal for segments that cross multiple cells.
5. **Write zarr.** The accumulated grid array (shape: latitude x longitude x
   altitude x species) is written to a zarr file with grid and filter metadata
   attached as attributes.

### Reduce phase

The reduce phase combines all map outputs into a single NetCDF file:

1. **Discover** all zarr files matching the `{map-prefix}-NNNNN.zarr` pattern.
   Validates that slice indices are contiguous starting from 0.
2. **Validate** that all slices have identical array shapes and consistent grid
   and filter metadata.
3. **Accumulate** by summing all slice arrays into a single float32 grid.
4. **Query** the mission database for the inventory time range.
5. **Write** the final CF-compliant NetCDF4 file via
   {py:class}`OutputGrid <AEIC.gridding.output.OutputGrid>`.

## Parallel execution

The `--slice-count` and `--slice-index` options split the trajectory store into
roughly equal chunks for parallel processing. Each map worker receives a
different `--slice-index` (0-based) and writes its own zarr file. After all map
jobs complete, a single reduce job combines them.

The slice calculation divides the total number of missions (or trajectories)
into `slice-count` groups of `ceil(N / slice-count)`, with the last slice
clamped to the remainder.

### Example with GNU parallel

Generate a task list and run 10 concurrent workers:

```shell
# Generate map tasks
NJOBS=100
for i in $(seq 0 $((NJOBS-1))); do
  echo "aeic trajectories-to-grid \
    --input-store output/trajectories.aeic-store \
    --grid-file src/AEIC/data/grids/era5.toml \
    --mode map \
    --map-prefix output/grid/map-slice \
    --slice-count $NJOBS \
    --slice-index $i"
done > tasks.txt

# Run with 10 concurrent workers
parallel -j 10 < tasks.txt

# Reduce all slices into the final output
aeic trajectories-to-grid \
  --input-store output/trajectories.aeic-store \
  --grid-file src/AEIC/data/grids/era5.toml \
  --mission-db-file oag-2019.sqlite \
  --mode reduce \
  --map-prefix output/grid/map-slice \
  --output-file output/inventory.nc
```

### SLURM job arrays

For HPC clusters, the same pattern maps naturally to SLURM job arrays. Set
`--slice-count` to the array size and `--slice-index` to `$SLURM_ARRAY_TASK_ID`
(0-based). Run the reduce step as a dependent job that waits for all map
tasks to complete.

## Output format

The reduce phase writes a CF-compliant NetCDF4 file with the following
structure.

### Dimensions

| Dimension | Size | Description |
|-----------|------|-------------|
| `time` | unlimited | Time steps (1 for annual output) |
| `latitude` | grid-dependent | Number of latitude bins |
| `longitude` | grid-dependent | Number of longitude bins |
| `altitude` or `pressure_level` | grid-dependent | Number of vertical bins |
| `nv` | 2 | Vertex count for bounds variables |

### Coordinate variables

- **`time`** — seconds since 1970-01-01 UTC. For annual output, contains a
  single value at the start of the inventory period.
- **`latitude`** — cell center values in degrees north, with `lat_bnds` bounds.
- **`longitude`** — cell center values in degrees east, with `lon_bnds` bounds.
- **`altitude`** (height grids) — cell center values in meters, `positive =
  'up'`, with `altitude_bnds` bounds.
- **`pressure_level`** (ISA pressure grids) — pressure values in hPa,
  `positive = 'down'`, stored in descending order (ERA5 convention).

### Species variables

One variable per chemical species (e.g. `co2`, `h2o`, `nox`), with dimensions
`(time, vertical, latitude, longitude)`. Values are in grams, stored as
float32 with zlib compression (level 4, shuffle enabled).

### Reproducibility provenance

The output file includes a `_reproducibility` group containing two subgroups:

- **`trajectory_generation`** — provenance from the input trajectory store
  (AEIC version, git state, configuration, files accessed, sampling parameters).
- **`gridding`** — provenance from the gridding run (AEIC version, git state,
  grid definition, filter expression, input paths, number of slices).

## API reference

### Grid classes

```{eval-rst}
.. autoclass:: AEIC.gridding.grid.Grid
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.gridding.grid.LatitudeGrid
   :members:
   :exclude-members: model_config
```

```{eval-rst}
.. autoclass:: AEIC.gridding.grid.LongitudeGrid
   :members:
   :exclude-members: model_config
```

```{eval-rst}
.. autoclass:: AEIC.gridding.grid.HeightGrid
   :members:
   :exclude-members: model_config
```

```{eval-rst}
.. autoclass:: AEIC.gridding.grid.ISAPressureGrid
   :members:
   :exclude-members: model_config
```

### Output

```{eval-rst}
.. autoclass:: AEIC.gridding.output.OutputGrid
   :members:
```
