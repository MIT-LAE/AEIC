# Grid map phase optimization report

## Summary

The `aeic trajectories-to-grid --mode map` command was optimized in
three steps, achieving a **47x overall speedup** on the map phase hot
loop.

| Commit | Change | map_phase time | Speedup |
|--------|--------|---------------|---------|
| (baseline) | Original code | 174.3 s | 1x |
| `9959fcb` | Eliminate redundant NetCDF reads | 32.2 s | 5.4x |
| `3eeb48e` | Batched slab reads via `iter_range` | 5.9 s | 29.3x |
| `3eeb48e^` | Skip type-checking on trusted internal reads | 3.7 s | 47.1x |

Benchmark: ~10,000 trajectories (slice 0 of 3900 from the test-run store),
1x1 degree grid, best of 3 runs, `map_phase` wall-clock only.


## Step 1: eliminate redundant scalar reads in `_read_from_nc_var`

**Commit:** `9959fcb`

### The problem

`_read_from_nc_var` reads a single field for a single trajectory from a
NetCDF variable. Several of its branches accessed `var[index]` more than
once for the same call:

```python
# Before (simplified)
case (False, False, False):
    if var[index] == var.get_fill_value():   # read 1
        return None
    return var[index]                        # read 2 (same data!)

case (True, False, False) | (True, False, True):
    return SpeciesValues(
        {sp: var[index, si] for si, sp in enumerate(species)}
        #     ^^^^^^^^^^^^^ one NetCDF read per species
    )
```

Each `var[index]` or `var[index, si]` is a full round-trip into the
netCDF4 C library: dimension lookup, type conversion, and (for HDF5) a
chunk decompression. With ~40 fields per trajectory and ~10,000
trajectories, the redundant reads added up to minutes of pure overhead.

### The fix

Read once into a local variable, then index into the in-memory result:

```python
# After
case (False, False, False):
    val = var[index]                         # one read
    if val == var.get_fill_value():
        return None
    return val

case (True, False, False) | (True, False, True):
    slab = var[index]                        # one read: shape (NSPECIES,)
    return SpeciesValues({sp: slab[si] for si, sp in enumerate(species)})
```


## Step 2: batched slab reads via `iter_range`

**Commit:** `3eeb48e`

### The problem

After Step 1, the remaining bottleneck was the per-trajectory call
overhead of `__getitem__` -> `_load_trajectory` -> `_read_from_nc_var`.
Each trajectory still required one NetCDF read per field, and the Python
function-call stack was entered ~40 times per trajectory. For 10,000
trajectories that is 400,000 individual NetCDF reads.

### The fix

`TrajectoryStore.iter_range(start, stop)` reads each field as a single
slab of `batch_size` trajectories (default 500), then unpacks the
in-memory array into individual `Trajectory` objects. This reduces
400,000 NetCDF reads to ~800 (40 fields x 20 batches).

Key design decisions:

- **File-boundary awareness.** For merged stores, each underlying
  `NcFiles` has a `size_index` marking where each constituent file ends.
  A single slab read cannot straddle a file boundary. `iter_range`
  collects boundary points from all `NcFiles`, splits the requested range
  at those boundaries, then chunks each sub-range by `batch_size`.

- **Positive local indices via `bisect_right`.** The existing
  `_load_trajectory` uses `bisect_left(size_index, index + 1)` and
  computes a *negative* local index (`index - size_index[file_index]`).
  This works for scalar reads because netCDF4 interprets negative scalar
  indices as "from the end." But for slab reads, `var[-2:-1]` uses
  Python slice semantics and produces wrong results. `iter_range` uses
  `bisect_right` and computes positive local offsets instead.

- **No cache interaction.** Trajectories from `iter_range` are not
  placed in the LRU cache (`self._trajectories`), since strict forward
  iteration would only thrash it.


## Step 3: skip type-checking on trusted internal reads

### The problem

After Step 2, profiling showed that the NetCDF slab reads themselves
(`_load_trajectories` own time) accounted for only 1.8 s of the 5.9 s
map phase. The new dominant cost was **trajectory construction**: the
`setattr` -> `Container.__setattr__` -> `convert_in` -> `_cast` chain
consumed 4.7 s validating and re-casting data that was already correctly
typed from the NetCDF read.

For each of the ~506,600 field assignments (50 fields x 10,131
trajectories), `_cast` calls `np.asarray`, `np.can_cast`, and
`ndarray.astype` -- all redundant when the data comes straight from a
NetCDF variable whose schema already enforces the correct type.

### The fix

In `_load_trajectories`, write directly to the private `_data` dict
instead of going through `__setattr__`:

```python
# Before
for k, v in per_traj_data[i].items():
    setattr(traj, k, v)          # -> convert_in -> _cast per field

# After
for k, v in per_traj_data[i].items():
    traj._data[k] = v            # trusted: data is from NetCDF slab read
```

This bypass is confined to the private `_load_trajectories` method.
The public `Container.__setattr__` type-checking is unchanged and
continues to protect all user-facing code paths.


## Lessons: avoiding NetCDF performance bugs

### 1. Every `var[index]` is expensive -- never read the same index twice

NetCDF4-Python's `Variable.__getitem__` is not a simple array lookup. It
enters the C library, locates the chunk, decompresses it, applies type
conversion, and returns a new Python/NumPy object. Accessing `var[i]`
twice does all of that work twice. Always capture the result in a local:

```python
# Bad
if var[i] == fill:
    return None
return var[i]

# Good
val = var[i]
if val == fill:
    return None
return val
```

This applies equally to multi-dimensional access like `var[i, j]`.

### 2. Prefer slab reads over scalar loops

Reading `var[start:stop]` is much cheaper than looping over
`var[i]` for `i in range(start, stop)`. The per-call overhead of
entering the C library dominates for small reads, so amortizing it
across a batch is a large win. The optimal batch size depends on memory
vs. overhead tradeoffs; 500 worked well here.

### 3. Watch out for negative indexing in slices

netCDF4-Python interprets a negative *scalar* index as "from the end of
the dimension," which happens to produce correct results when the local
offset within a merged file is negative by construction. But negative
indices in *slices* (`var[-3:-1]`) follow Python's slice-from-the-end
convention, which silently returns wrong data. If you are computing
local offsets for slab reads, make sure they are always non-negative.

### 4. Profile before optimizing

The original plan included parallelizing the Numba gridding kernel. A
cProfile run showed it accounted for less than 1 second of 32 seconds --
the I/O path was 30x more important. Always profile to find the actual
bottleneck before writing optimization code.

```bash
uv run python -m cProfile -o /tmp/profile.prof \
    .venv/bin/aeic trajectories-to-grid <args>

python -c "
import pstats
s = pstats.Stats('/tmp/profile.prof')
s.sort_stats('tottime').print_stats(40)
"
```

### 5. Validation has a cost -- bypass it on trusted internal paths

Defensive type-checking in `__setattr__` is valuable for user-facing
APIs, but when data is being loaded from a schema-enforced store (like
NetCDF), the types are already correct by construction. In tight loops,
the per-assignment overhead of `np.can_cast` + `ndarray.astype` adds up
quickly. If you can identify an internal code path where the data
provenance is guaranteed, writing directly to the underlying storage
(`_data` dict) avoids redundant validation. Keep such bypasses private
and well-commented so they don't leak into user-facing code.

### 6. Understand the library stack beneath you

netCDF4-Python wraps netCDF-C, which wraps HDF5. Each layer adds
overhead per call. Patterns that look fine in pure Python (dictionary
comprehensions that index a variable N times) become expensive when each
index operation traverses three library boundaries. Think of NetCDF
variable access as I/O, not as array indexing.
