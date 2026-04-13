import json
import logging
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import netCDF4 as nc4
import numpy as np

from AEIC.gridding.grid import Grid, HeightGrid, ISAPressureGrid
from AEIC.storage.reproducibility import (
    GIT_BRANCH,
    GIT_COMMIT,
    GIT_DIRTY,
    VERSION,
    ReproducibilityData,
)
from AEIC.types import Species

logger = logging.getLogger(__name__)


@dataclass
class OutputGrid:
    """Accumulated gridded emissions ready to be written to a NetCDF file.

    ``accum`` is a float32 array with shape ``(nlat, nlon, nalt, nspecies)``
    produced by the map phase. ``write()`` converts it to a CF-compliant
    NetCDF4 inventory file with coordinate variables, per-species emissions
    variables, and structured reproducibility provenance groups.
    """

    grid: Grid
    species: list[Species]
    accum: np.ndarray
    min_ts: float
    n_slices: int
    input_store: Path
    mission_db_file: Path
    traj_repro: ReproducibilityData | None
    traj_comments: list[str]
    filter_json: str | None

    def write(self, output_file: Path) -> None:
        """Write the accumulated gridded data to a NetCDF file.

        Always passes ``keepweakref=True`` per the netcdf4-python issue #1444
        workaround used elsewhere in this repo. The accumulator is laid out as
        ``(lat, lon, alt, species)`` and each species slab is permuted to
        ``(alt, lat, lon)`` when written.
        """
        with nc4.Dataset(
            str(output_file), mode='w', format='NETCDF4', keepweakref=True
        ) as ds:
            # Dimensions: time is unlimited so monthly (length 12) can be added
            # later without breaking the schema. The vertical dimension is
            # created below in the per-grid-type dispatch (altitude or
            # pressure_level).
            ds.createDimension('time', None)
            ds.createDimension('latitude', self.grid.latitude.bins)
            ds.createDimension('longitude', self.grid.longitude.bins)
            ds.createDimension('nv', 2)

            # Time coordinate variable. The annual case writes a single value
            # at the start of the inventory period.
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = 'seconds since 1970-01-01 00:00:00 UTC'
            time_var.calendar = 'gregorian'
            time_var.standard_name = 'time'
            time_var.long_name = 'Start of inventory period'
            time_var[0] = float(self.min_ts)

            # Latitude / longitude coordinate variables (cell centers + bounds).
            lat = self.grid.latitude
            lon = self.grid.longitude
            lat_centers = lat.range[0] + (np.arange(lat.bins) + 0.5) * lat.resolution
            lon_centers = lon.range[0] + (np.arange(lon.bins) + 0.5) * lon.resolution
            lat_edges = lat.range[0] + np.arange(lat.bins + 1) * lat.resolution
            lon_edges = lon.range[0] + np.arange(lon.bins + 1) * lon.resolution

            lat_var = ds.createVariable('latitude', 'f8', ('latitude',))
            lat_var.units = 'degrees_north'
            lat_var.standard_name = 'latitude'
            lat_var.bounds = 'lat_bnds'
            lat_var[:] = lat_centers

            lat_bnds_var = ds.createVariable('lat_bnds', 'f8', ('latitude', 'nv'))
            lat_bnds_var[:, 0] = lat_edges[:-1]
            lat_bnds_var[:, 1] = lat_edges[1:]

            lon_var = ds.createVariable('longitude', 'f8', ('longitude',))
            lon_var.units = 'degrees_east'
            lon_var.standard_name = 'longitude'
            lon_var.bounds = 'lon_bnds'
            lon_var[:] = lon_centers

            lon_bnds_var = ds.createVariable('lon_bnds', 'f8', ('longitude', 'nv'))
            lon_bnds_var[:, 0] = lon_edges[:-1]
            lon_bnds_var[:, 1] = lon_edges[1:]

            # Altitude / pressure coordinate variable (per-mode dispatch).
            alt = self.grid.altitude
            if isinstance(alt, HeightGrid):
                ds.createDimension('altitude', alt.bins)
                alt_var = ds.createVariable('altitude', 'f8', ('altitude',))
                alt_var.units = 'm'
                alt_var.standard_name = 'altitude'
                alt_var.positive = 'up'
                alt_var.bounds = 'altitude_bnds'
                alt_var[:] = alt.levels

                alt_edges = alt.edges
                alt_bnds_var = ds.createVariable(
                    'altitude_bnds', 'f8', ('altitude', 'nv')
                )
                alt_bnds_var[:, 0] = alt_edges[:-1]
                alt_bnds_var[:, 1] = alt_edges[1:]
                vert_dim = 'altitude'
            elif isinstance(alt, ISAPressureGrid):
                ds.createDimension('pressure_level', alt.bins)
                pl_var = ds.createVariable('pressure_level', 'f8', ('pressure_level',))
                pl_var.units = 'hPa'
                pl_var.long_name = 'pressure'
                pl_var.standard_name = 'air_pressure'
                pl_var.positive = 'down'
                pl_var.stored_direction = 'decreasing'
                # Levels stored in descending order (ERA5 convention).
                pl_var[:] = np.sort(alt.levels)[::-1]
                vert_dim = 'pressure_level'
            else:
                raise NotImplementedError(
                    f'Unsupported altitude grid type: {type(alt).__name__}.'
                )

            # Per-species emissions variables. Units come from EMISSIONS_FIELDS
            # in AEIC.emissions.emission (trajectory_emissions is in grams).
            # The accumulator is laid out as (lat, lon, vert, species). After
            # transposing to (vert, lat, lon), pressure grids need a vertical
            # flip so the output matches the descending-pressure ERA5 convention
            # (the kernel bins in ascending pressure order).
            for i, sp in enumerate(self.species):
                var = ds.createVariable(
                    sp.name.lower(),
                    'f4',
                    ('time', vert_dim, 'latitude', 'longitude'),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                )
                var.units = 'g'
                var.description = f'Gridded {sp.name} emissions'
                slab = self.accum[..., i].transpose(2, 0, 1)
                if isinstance(alt, ISAPressureGrid):
                    slab = slab[::-1, :, :]
                var[0, :, :, :] = slab

            # Minimal global attributes for quick identification.
            ds.aeic_version = VERSION
            ds.created_utc = datetime.now(UTC).isoformat()

            # Reproducibility groups under _reproducibility/.
            repro = ds.createGroup('_reproducibility')

            # -- Trajectory generation provenance (from the input store).
            if self.traj_repro is not None:
                tg = repro.createGroup('trajectory_generation')
                tg.createVariable('python_version', str, ())
                tg.variables['python_version'][...] = self.traj_repro.python_version
                tg.createVariable('aeic_version', str, ())
                tg.variables['aeic_version'][...] = self.traj_repro.software_version
                tg.createVariable('git_commit', str, ())
                tg.variables['git_commit'][...] = (
                    self.traj_repro.git_commit
                    if self.traj_repro.git_commit is not None
                    else ''
                )
                tg.createVariable('git_branch', str, ())
                tg.variables['git_branch'][...] = (
                    self.traj_repro.git_branch
                    if self.traj_repro.git_branch is not None
                    else ''
                )
                tg.createVariable('git_dirty', np.uint8, ())
                tg.variables['git_dirty'][...] = self.traj_repro.git_dirty
                tg.createVariable('files_accessed', str, ())
                tg.variables['files_accessed'][...] = json.dumps(
                    [str(p) for p in self.traj_repro.files]
                )
                tg.createVariable('config', str, ())
                tg.variables['config'][...] = self.traj_repro.config
                tg.createVariable('comments', str, ())
                tg.variables['comments'][...] = json.dumps(self.traj_comments)
                if self.traj_repro.sample_fraction is not None:
                    tg.createVariable('sample_fraction', np.float64, ())
                    tg.variables['sample_fraction'][...] = (
                        self.traj_repro.sample_fraction
                    )
                if self.traj_repro.sample_seed is not None:
                    tg.createVariable('sample_seed', np.int64, ())
                    tg.variables['sample_seed'][...] = self.traj_repro.sample_seed

            # -- Gridding provenance (from this run).
            gg = repro.createGroup('gridding')
            gg.createVariable('aeic_version', str, ())
            gg.variables['aeic_version'][...] = VERSION
            gg.createVariable('python_version', str, ())
            gg.variables['python_version'][...] = platform.python_version()
            gg.createVariable('git_commit', str, ())
            gg.variables['git_commit'][...] = (
                GIT_COMMIT if GIT_COMMIT is not None else ''
            )
            gg.createVariable('git_branch', str, ())
            gg.variables['git_branch'][...] = (
                GIT_BRANCH if GIT_BRANCH is not None else ''
            )
            gg.createVariable('git_dirty', np.uint8, ())
            gg.variables['git_dirty'][...] = GIT_DIRTY
            gg.createVariable('grid_definition', str, ())
            gg.variables['grid_definition'][...] = self.grid.model_dump_json()
            if self.filter_json is not None:
                gg.createVariable('filter', str, ())
                gg.variables['filter'][...] = self.filter_json
            gg.createVariable('input_store', str, ())
            gg.variables['input_store'][...] = str(Path(self.input_store).resolve())
            gg.createVariable('mission_db_file', str, ())
            gg.variables['mission_db_file'][...] = str(
                Path(self.mission_db_file).resolve()
            )
            gg.createVariable('n_slices', np.int32, ())
            gg.variables['n_slices'][...] = self.n_slices
            gg.createVariable('created_utc', str, ())
            gg.variables['created_utc'][...] = datetime.now(UTC).isoformat()

        logger.info('Wrote gridded inventory to %s', output_file)
