# CLI reference

AEIC installs a single entry-point command, `aeic`, which exposes six
subcommands covering the end-to-end inventory workflow:

| Subcommand | Purpose |
|------------|---------|
| `aeic convert-oag-data` | Convert an OAG CSV file to a mission database. See [OAG flight data](oag.md). |
| `aeic make-performance-model` | Build a performance model TOML file. See [Performance model files](performance_models/performance_model_files.md). |
| `aeic run` | Simulate trajectories for the missions in a database. See [below](#aeic-run). |
| `aeic merge-stores` | Merge per-slice trajectory stores produced by parallel simulation runs. See [below](#aeic-merge-stores). |
| `aeic make-file-bundle` | Build a reproducibility bundle containing every file referenced by a trajectory store. See [below](#aeic-make-file-bundle). |
| `aeic trajectories-to-grid` | Convert a trajectory store into a gridded NetCDF inventory. See [Gridding module](gridding.md). |

```{note}
The full flag reference for each subcommand lives in one place:
`aeic run`, `aeic merge-stores`, and `aeic make-file-bundle` are
documented inline below; `aeic convert-oag-data`,
`aeic make-performance-model`, and `aeic trajectories-to-grid` have
flag tables on their dedicated topic pages, linked from the table
above.
```

## `aeic run`

Run trajectory simulations for every mission returned by a mission database
query and write the results to a trajectory store. Simulations can be split
into slices so that the work can be spread across parallel processes (one
process per slice, each writing its own store file). The per-slice stores can
then be merged with {ref}`aeic merge-stores <aeic-merge-stores>` before
gridding.

### Command-line options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--config-file` | Path | Yes | | AEIC configuration TOML file. |
| `--performance-selector-dir` | Path | See note | | Directory of performance model files, one per mission. |
| `--performance-model-file` | Path | See note | | Single performance model TOML file to use for all missions. |
| `--mission-db-file` | Path | Yes | | Mission database (SQLite) to source flights from. |
| `--output-store` | Path | Yes | | Output trajectory store path prefix. The slice index is appended to form the actual file name. |
| `--sample` | Float in `[0, 1]` | No | | Fraction of missions to simulate. |
| `--seed` | Integer | No | | Random seed used when sampling missions. |
| `--slice-count` | Integer | No | `1` | Total number of parallel slices. |
| `--slice-index` | Integer | No | `0` | Zero-based index of the slice to process. |

```{note}
Exactly one of `--performance-selector-dir` or `--performance-model-file` must
be provided. The selector form picks a different performance model per
mission; the single-file form uses the same model for every mission.
```

### Single-process example

```shell
aeic run \
  --config-file aeic.toml \
  --performance-model-file models/B738.toml \
  --mission-db-file oag-2024.sqlite \
  --output-store output/trajectories
```

### Parallel example

Each parallel worker passes a different `--slice-index` (0-based) and writes
its own per-slice store file. The resulting files can then be merged with
`aeic merge-stores`.

```shell
NJOBS=100
for i in $(seq 0 $((NJOBS-1))); do
  echo "aeic run \
    --config-file aeic.toml \
    --performance-model-file models/B738.toml \
    --mission-db-file oag-2024.sqlite \
    --output-store output/trajectories \
    --slice-count $NJOBS \
    --slice-index $i"
done > run-tasks.txt

parallel -j 10 < run-tasks.txt

aeic merge-stores \
  --output-store output/trajectories.aeic-store \
  output/trajectories-*.nc
```

(aeic-merge-stores)=
## `aeic merge-stores`

Combine multiple per-slice trajectory stores into a single store. This is the
companion to the parallel mode of `aeic run`: each slice writes its own
NetCDF file, and `merge-stores` stitches them together so that downstream
steps (for example `aeic trajectories-to-grid`) see a single logical store.

### Command-line options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--output-store` | Path | Yes | | Path to the output store to create. |
| `--merge` / `--combine` | Flag | No | `--merge` | Controls the merge strategy (see below). |
| `INPUT_STORES` | Paths (positional, N-ary) | Yes | | One or more per-slice store files to merge. |

```{note}
`--merge` (the default) creates a **multi-file** store: the output store
references the input files in place, so they must remain on disk. `--combine`
copies the data into a **single-file** store which is self-contained but
larger. Prefer `--merge` for large inventories and `--combine` when the
output needs to be moved between machines.
```

### Example

```shell
aeic merge-stores \
  --output-store output/trajectories.aeic-store \
  output/trajectories-*.nc
```

(aeic-make-file-bundle)=
## `aeic make-file-bundle`

Produce a zip archive containing every file referenced by a trajectory
store's reproducibility data (configuration, performance model, mission
database, grid definitions, ...). The bundle, together with the trajectory
store itself, is sufficient to reproduce a downstream gridding or analysis
run.

The store must have been created with file-access tracking enabled; if the
store has no reproducibility data the command logs an error and exits
without producing a bundle.

### Command-line options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--input-store` | Path | Yes | Input trajectory store referencing the files to bundle. |
| `--output-bundle` | Path | Yes | Output zip file path. |

### Example

```shell
aeic make-file-bundle \
  --input-store output/trajectories.aeic-store \
  --output-bundle output/trajectories-bundle.zip
```
