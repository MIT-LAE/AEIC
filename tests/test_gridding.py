import sqlite3

import click
import netCDF4 as nc4
import numpy as np
import pytest
import zarr

from AEIC.commands.trajectories_to_grid import reduce_phase
from AEIC.config import config
from AEIC.gridding.grid import Grid
from AEIC.types import Species


def test_height_grid_load():
    # Load a grid with heights in the vertical and check values.
    grid = Grid.load(config.file_location('grids/basic-1x1.toml'))
    assert grid.longitude.resolution == 1.0
    assert grid.longitude.offset == 0.5
    assert grid.latitude.resolution == 1.0
    assert grid.latitude.offset == 0.5
    assert grid.altitude.mode == 'height'
    assert grid.altitude.resolution == 500.0
    assert grid.altitude.bottom == 0.0
    assert not hasattr(grid.altitude, 'levels')


def test_pressure_grid_load():
    # Load a grid with ISA pressures in the vertical and check values.
    grid = Grid.load(config.file_location('grids/basic-1x1-isa-pressure.toml'))
    assert grid.longitude.resolution == 1.0
    assert grid.longitude.offset == 0.5
    assert grid.latitude.resolution == 1.0
    assert grid.latitude.offset == 0.5
    assert grid.altitude.mode == 'isa_pressure'
    assert grid.altitude.levels == [1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0]
    assert grid.altitude.bottom == 1000.0
    assert grid.altitude.top == 100.0


# ---------------------------------------------------------------------------
# Helpers for reduce-phase tests
# ---------------------------------------------------------------------------


def _make_mission_db(path, timestamps):
    """Create a minimal SQLite mission DB with the given departure timestamps."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        'CREATE TABLE schedules (id INTEGER PRIMARY KEY, departure_timestamp INTEGER)'
    )
    for ts in timestamps:
        conn.execute('INSERT INTO schedules (departure_timestamp) VALUES (?)', (ts,))
    conn.commit()
    conn.close()


def _write_zarr_slice(path, data):
    """Write a numpy array to a zarr store at the given path."""
    arr = zarr.create_array(store=str(path), dtype='f4', shape=data.shape)
    arr[:] = data


# ---------------------------------------------------------------------------
# Happy-path test
# ---------------------------------------------------------------------------


def test_reduce_phase_happy_path(tmp_path):
    grid_file = config.file_location('grids/basic-4x4.toml')
    grid = Grid.load(grid_file)
    species = [Species.CO2, Species.H2O]
    nspecies = len(species)
    shape = grid.shape + (nspecies,)  # (nlat, nlon, nalt, nspecies)

    # Create two zarr slice files: slice 0 is all 2.0, slice 1 is all 3.0.
    map_prefix = str(tmp_path / 'map')
    _write_zarr_slice(f'{map_prefix}-00000.zarr', np.full(shape, 2.0, dtype=np.float32))
    _write_zarr_slice(f'{map_prefix}-00001.zarr', np.full(shape, 3.0, dtype=np.float32))

    # Minimal mission DB: two timestamps spanning 2019.
    db_path = tmp_path / 'missions.sqlite'
    _make_mission_db(db_path, [1546300800, 1577836799])  # 2019-01-01, 2019-12-31

    output_file = tmp_path / 'out.nc'
    reduce_phase(
        grid,
        species,
        map_prefix,
        output_file,
        db_path,
        tmp_path / 'fake_store.nc',
        grid_file,
    )

    with nc4.Dataset(str(output_file), keepweakref=True) as ds:
        # Dimensions
        assert ds.dimensions['time'].size == 1
        assert ds.dimensions['altitude'].size == grid.altitude.bins
        assert ds.dimensions['latitude'].size == grid.latitude.bins
        assert ds.dimensions['longitude'].size == grid.longitude.bins

        # Each cell was filled with 2.0 + 3.0 = 5.0 across all slices.
        # Sum over (alt, lat, lon) = 5.0 * nlat * nlon * nalt.
        n_cells = grid.latitude.bins * grid.longitude.bins * grid.altitude.bins
        expected_sum = 5.0 * n_cells
        for sp in species:
            varname = sp.name.lower()
            assert varname in ds.variables, f'Missing variable {varname}'
            var = ds.variables[varname]
            assert var.units == 'g'
            assert np.isclose(float(var[0, :, :, :].sum()), expected_sum, rtol=1e-4)

        # Global attributes
        assert hasattr(ds, 'aeic_version')
        assert hasattr(ds, 'created_utc')
        assert ds.n_slices == 2
        assert len(ds.grid_definition) > 0

        # Latitude cell centers
        nlat = grid.latitude.bins
        nlon = grid.longitude.bins
        lat_centers = (
            grid.latitude.range[0] + (np.arange(nlat) + 0.5) * grid.latitude.resolution
        )
        lon_centers = (
            grid.longitude.range[0]
            + (np.arange(nlon) + 0.5) * grid.longitude.resolution
        )
        assert np.allclose(ds.variables['latitude'][:], lat_centers)
        assert np.allclose(ds.variables['longitude'][:], lon_centers)

        # Bounds variables present
        assert 'lat_bnds' in ds.variables
        assert 'lon_bnds' in ds.variables


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_reduce_phase_empty_glob(tmp_path):
    # No zarr slices found → UsageError.
    grid_file = config.file_location('grids/basic-4x4.toml')
    grid = Grid.load(grid_file)
    db_path = tmp_path / 'missions.sqlite'
    _make_mission_db(db_path, [1546300800])

    with pytest.raises(click.UsageError, match='No slice files'):
        reduce_phase(
            grid,
            [Species.CO2],
            str(tmp_path / 'map'),
            tmp_path / 'out.nc',
            db_path,
            tmp_path / 'fake.nc',
            grid_file,
        )


def test_reduce_phase_missing_slice_index(tmp_path):
    # Gap in slice indices → UsageError mentioning contiguous.
    grid_file = config.file_location('grids/basic-4x4.toml')
    grid = Grid.load(grid_file)
    species = [Species.CO2]
    shape = grid.shape + (1,)
    map_prefix = str(tmp_path / 'map')

    # Write slices 0 and 2 but omit 1.
    for idx in (0, 2):
        _write_zarr_slice(
            f'{map_prefix}-{idx:05d}.zarr',
            np.zeros(shape, dtype=np.float32),
        )

    db_path = tmp_path / 'missions.sqlite'
    _make_mission_db(db_path, [1546300800])

    with pytest.raises(click.UsageError, match='contiguous'):
        reduce_phase(
            grid,
            species,
            map_prefix,
            tmp_path / 'out.nc',
            db_path,
            tmp_path / 'fake.nc',
            grid_file,
        )


def test_reduce_phase_wrong_shape(tmp_path):
    # Second slice has wrong shape → UsageError mentioning shape.
    grid_file = config.file_location('grids/basic-4x4.toml')
    grid = Grid.load(grid_file)
    species = [Species.CO2]
    correct_shape = grid.shape + (1,)
    wrong_shape = grid.shape + (99,)  # wrong species count
    map_prefix = str(tmp_path / 'map')

    _write_zarr_slice(
        f'{map_prefix}-00000.zarr', np.zeros(correct_shape, dtype=np.float32)
    )
    _write_zarr_slice(
        f'{map_prefix}-00001.zarr', np.zeros(wrong_shape, dtype=np.float32)
    )

    db_path = tmp_path / 'missions.sqlite'
    _make_mission_db(db_path, [1546300800])

    with pytest.raises(click.UsageError, match='shape'):
        reduce_phase(
            grid,
            species,
            map_prefix,
            tmp_path / 'out.nc',
            db_path,
            tmp_path / 'fake.nc',
            grid_file,
        )
