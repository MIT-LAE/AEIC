import importlib.metadata
import logging
import math
import re
import time
import tomllib
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import click
import netCDF4 as nc4
import numpy as np
import zarr

from AEIC.gridding.grid import Grid, HeightGrid, ISAPressureGrid
from AEIC.gridding.kernels import process_segments_nonuniform_z
from AEIC.missions import CountQuery, Database, Filter, Query, TimeRangeQuery
from AEIC.trajectories import Trajectory, TrajectoryStore
from AEIC.types import Species
from AEIC.utils.progress import Progress
from AEIC.utils.standard_atmosphere import pressure_at_altitude_isa_bada4

logger = logging.getLogger(__name__)


NSEGMENTS = 1000  # Number of segments to process in each chunk.


def map_phase(
    ntrajs: int,
    species: list[Species],
    traj_iter: Generator[Trajectory],
    grid: Grid,
    map_output: str,
):
    # Build output array. This assumes that all trajectories have the same set
    # of species, which should be the case if they are all processed with the
    # same configuration.
    output = np.zeros(grid.shape + (len(species),))

    # Accumulate segments in batches: arrays for segment endpoints and
    # emissions are reused across batches. Emissions are recorded at the first
    # point of the segment, consistent with the way they are recorded in the
    # trajectory data.
    lat0 = np.zeros(NSEGMENTS)
    lon0 = np.zeros(NSEGMENTS)
    z0 = np.zeros(NSEGMENTS)
    lat1 = np.zeros(NSEGMENTS)
    lon1 = np.zeros(NSEGMENTS)
    z1 = np.zeros(NSEGMENTS)
    emissions = np.zeros((NSEGMENTS, len(species)))

    # Get vertical grid edges.
    z_edges = grid.altitude.edges
    use_pressure = isinstance(grid.altitude, ISAPressureGrid)

    # Number of segments currently accumulated in the arrays.
    nsegs = 0

    p = Progress(total=ntrajs, desc='Trajectory')
    lat_min = min(grid.latitude.range)
    lon_min = min(grid.longitude.range)
    for traj in traj_iter:
        # Split trajectories across the date line and process each
        # sub-trajectory separately.
        for sub_traj in traj.dateline_split():
            # If adding the segments from this sub-trajectory would exceed the
            # batch size, process the current batch and start a new one.
            if nsegs + len(sub_traj) > NSEGMENTS:
                process_segments_nonuniform_z(
                    lat0[:nsegs],
                    lon0[:nsegs],
                    z0[:nsegs],
                    lat1[:nsegs],
                    lon1[:nsegs],
                    z1[:nsegs],
                    emissions[:nsegs, :],
                    output,
                    z_edges,
                    lat_min,
                    grid.latitude.resolution,
                    lon_min,
                    grid.longitude.resolution,
                )
                nsegs = 0

            # Add segments from this sub-trajectory to the batch arrays.
            start = nsegs
            end = nsegs + len(sub_traj) - 1
            lat0[start:end] = sub_traj.latitude[:-1]
            lon0[start:end] = sub_traj.longitude[:-1]
            lat1[start:end] = sub_traj.latitude[1:]
            lon1[start:end] = sub_traj.longitude[1:]
            if use_pressure:
                alt0 = sub_traj.altitude[:-1]
                alt1 = sub_traj.altitude[1:]
                z0[start:end] = pressure_at_altitude_isa_bada4(alt0) / 100.0
                z1[start:end] = pressure_at_altitude_isa_bada4(alt1) / 100.0
            else:
                z0[start:end] = sub_traj.altitude[:-1]
                z1[start:end] = sub_traj.altitude[1:]
            for i, sp in enumerate(species):
                emissions[start:end, i] = sub_traj.trajectory_emissions[sp][:-1]
            nsegs += len(sub_traj) - 1

        p.update()
    p.close()

    # Process any left-over segments in the batch arrays.
    if nsegs > 0:
        process_segments_nonuniform_z(
            lat0[:nsegs],
            lon0[:nsegs],
            z0[:nsegs],
            lat1[:nsegs],
            lon1[:nsegs],
            z1[:nsegs],
            emissions[:nsegs, :],
            output,
            z_edges,
            lat_min,
            grid.latitude.resolution,
            lon_min,
            grid.longitude.resolution,
        )

    # Save output array to zarr file all in one go: more efficient than
    # incrementally updating the zarr file for each batch.
    save = zarr.create_array(store=map_output, dtype='f4', shape=output.shape)
    save[:] = output


def _discover_slice_files(map_prefix: str) -> tuple[dict[int, Path], list[int]]:
    """Discover and validate slice zarr files matching the map_prefix pattern.

    Returns a dict mapping slice index to path, and the sorted list of indices.
    Raises ``click.UsageError`` if no files are found or the indices are not
    contiguous starting from 0.
    """
    prefix_path = Path(map_prefix)
    parent = prefix_path.parent if str(prefix_path.parent) else Path('.')
    base_name = prefix_path.name
    pattern = re.compile(rf'^{re.escape(base_name)}-(\d{{5}})\.zarr$')

    slice_files: dict[int, Path] = {}
    if parent.exists():
        for entry in parent.iterdir():
            m = pattern.match(entry.name)
            if m is not None:
                slice_files[int(m.group(1))] = entry

    if not slice_files:
        raise click.UsageError(
            f'No slice files matching {base_name}-NNNNN.zarr found in {parent}.'
        )

    indices = sorted(slice_files.keys())
    expected_indices = list(range(len(indices)))
    if indices != expected_indices:
        missing = sorted(set(expected_indices) - set(indices))
        raise click.UsageError(
            f'Slice files are not contiguous: found {len(indices)} files but '
            f'expected indices 0..{len(indices) - 1}. Missing: {missing}.'
        )

    logger.info('Found %d slice file(s) under %s', len(indices), parent)
    return slice_files, indices


def _validate_slice_shapes(
    slice_files: dict[int, Path],
    indices: list[int],
    expected_shape: tuple[int, ...],
) -> None:
    """Check every slice zarr has the expected shape.

    The map array layout is (lat, lon, alt, species). Raises
    ``click.UsageError`` on the first mismatch.
    """
    first_arr = zarr.open_array(store=str(slice_files[0]), mode='r')
    if tuple(first_arr.shape) != expected_shape:
        raise click.UsageError(
            f'Slice file {slice_files[0]} has shape {tuple(first_arr.shape)}, '
            f'expected {expected_shape} from --grid-file and --input-store. '
            f'Did you pass the same grid file used during the map phase?'
        )
    for i in indices[1:]:
        arr = zarr.open_array(store=str(slice_files[i]), mode='r')
        if tuple(arr.shape) != expected_shape:
            raise click.UsageError(
                f'Slice file {slice_files[i]} has shape {tuple(arr.shape)} '
                f'which differs from the first slice {expected_shape}.'
            )


def _accumulate_slices(
    slice_files: dict[int, Path],
    indices: list[int],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Sum all slice zarr arrays into a single float32 accumulator."""
    accum = np.zeros(expected_shape, dtype=np.float32)
    p = Progress(total=len(indices), desc='Slice')
    for i in indices:
        arr = zarr.open_array(store=str(slice_files[i]), mode='r')
        accum += arr[:]  # type: ignore
        p.update()
    p.close()
    return accum


def _query_inventory_time_range(
    mission_db_file: Path,
) -> tuple[float, datetime, datetime]:
    """Query the mission DB for the earliest and latest scheduled departure.

    Returns ``(min_ts, period_start, period_end)`` where ``min_ts`` is the
    Unix timestamp of the earliest departure and the datetimes are UTC.
    Raises ``click.UsageError`` if the database contains no flights.
    """
    with Database(mission_db_file) as db:
        time_range = db(TimeRangeQuery())
    assert isinstance(time_range, tuple)
    min_ts, max_ts = time_range
    if min_ts is None or max_ts is None:
        raise click.UsageError(
            f'Mission database {mission_db_file} contains no scheduled '
            f'flights, so a time coordinate cannot be derived.'
        )
    period_start = datetime.fromtimestamp(min_ts, tz=UTC)
    period_end = datetime.fromtimestamp(max_ts, tz=UTC)
    logger.info('Inventory period: %s to %s', period_start, period_end)
    return min_ts, period_start, period_end


def _write_netcdf_output(
    output_file: Path,
    grid: Grid,
    species: list[Species],
    accum: np.ndarray,
    min_ts: float,
    n_slices: int,
    input_store: Path,
    mission_db_file: Path,
    grid_file: Path,
) -> None:
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
        # later without breaking the schema. The vertical dimension is created
        # below in the per-grid-type dispatch (altitude or pressure_level).
        ds.createDimension('time', None)
        ds.createDimension('latitude', grid.latitude.bins)
        ds.createDimension('longitude', grid.longitude.bins)
        ds.createDimension('nv', 2)

        # Time coordinate variable. The annual case writes a single value
        # at the start of the inventory period.
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.units = 'seconds since 1970-01-01 00:00:00 UTC'
        time_var.calendar = 'gregorian'
        time_var.standard_name = 'time'
        time_var.long_name = 'Start of inventory period'
        time_var[0] = float(min_ts)

        # Latitude / longitude coordinate variables (cell centers + bounds).
        lat = grid.latitude
        lon = grid.longitude
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
        alt = grid.altitude
        if isinstance(alt, HeightGrid):
            ds.createDimension('altitude', alt.bins)
            alt_var = ds.createVariable('altitude', 'f8', ('altitude',))
            alt_var.units = 'm'
            alt_var.standard_name = 'altitude'
            alt_var.positive = 'up'
            alt_var.bounds = 'altitude_bnds'
            alt_var[:] = alt.levels

            alt_edges = alt.edges
            alt_bnds_var = ds.createVariable('altitude_bnds', 'f8', ('altitude', 'nv'))
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

        # Per-species emissions variables. Units come from EMISSIONS_FIELDS in
        # AEIC.emissions.emission (trajectory_emissions is in grams).
        # The accumulator is laid out as (lat, lon, vert, species). After
        # transposing to (vert, lat, lon), pressure grids need a vertical
        # flip so the output matches the descending-pressure ERA5 convention
        # (the kernel bins in ascending pressure order).
        for i, sp in enumerate(species):
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
            slab = accum[..., i].transpose(2, 0, 1)
            if isinstance(alt, ISAPressureGrid):
                slab = slab[::-1, :, :]
            var[0, :, :, :] = slab

        # Global attributes for provenance and reproducibility.
        try:
            aeic_version = importlib.metadata.version('AEIC')
        except importlib.metadata.PackageNotFoundError:
            aeic_version = 'unknown'
        ds.aeic_version = aeic_version
        ds.created_utc = datetime.now(UTC).isoformat()
        ds.input_store = str(Path(input_store).resolve())
        ds.mission_db_file = str(Path(mission_db_file).resolve())
        ds.grid_file = str(Path(grid_file).resolve())
        ds.n_slices = n_slices
        ds.grid_definition = Path(grid_file).read_text()

    logger.info('Wrote gridded inventory to %s', output_file)


def reduce_phase(
    grid: Grid,
    species: list[Species],
    map_prefix: str,
    output_file: Path,
    mission_db_file: Path,
    input_store: Path,
    grid_file: Path,
):
    """Combine map-phase zarr slice files into a single gridded NetCDF file.

    The map phase writes one bare zarr array per slice, named
    ``{map_prefix}-NNNNN.zarr`` (where ``NNNNN`` is the zero-padded slice
    index), each with shape ``(nlat, nlon, nalt, nspecies)`` and dtype
    ``f4``. This function discovers all such slice files under
    ``map_prefix``, sums them into a single accumulator, queries the mission
    database for the inventory time range, and writes a NetCDF file
    containing one variable per species plus CF-style coordinate variables.
    """
    slice_files, indices = _discover_slice_files(map_prefix)
    n_slices = len(indices)

    expected_shape = (
        grid.latitude.bins,
        grid.longitude.bins,
        grid.altitude.bins,
        len(species),
    )
    _validate_slice_shapes(slice_files, indices, expected_shape)

    accum = _accumulate_slices(slice_files, indices, expected_shape)

    min_ts, _, _ = _query_inventory_time_range(mission_db_file)

    _write_netcdf_output(
        output_file,
        grid,
        species,
        accum,
        min_ts,
        n_slices,
        input_store,
        mission_db_file,
        grid_file,
    )


@click.command(
    help="""Convert trajectory data to gridded format for analysis
and visualization.

This works in two phases: map and reduce. In map mode, the input trajectory
store is processed in chunks, and intermediate grid files are saved as zarr
files. In reduce mode, the intermediate zarr files are read and combined into a
final output NetCDF file."""
)
@click.option(
    '--input-store',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Input trajectory store.',
)
@click.option(
    '--mission-db-file',
    type=click.Path(exists=True, path_type=Path),
    help='Mission database file.',
)
@click.option(
    '--filter-file',
    type=click.Path(exists=True, path_type=Path),
    help='Trajectory filter definition file.',
)
@click.option(
    '--grid-file',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Grid definition file.',
)
@click.option(
    '--mode',
    type=click.Choice(['map', 'reduce'], case_sensitive=False),
    required=True,
    help='Processing mode: map or reduce.',
)
# TODO: Add species option to filter for specific species in the output.
@click.option(
    '--output-times',
    type=click.Choice(['annual', 'monthly', 'daily'], case_sensitive=False),
    default='annual',
    help='Output time resolution (default: annual).',
)
@click.option(
    '--output-file',
    type=click.Path(path_type=Path),
    help='Final NetCDF output file path (required in reduce and map-reduce mode).',
)
@click.option(
    '--map-prefix',
    required=True,
    help='Map phase intermediate output file prefix.',
)
@click.option(
    '--slice-count',
    type=int,
    default=1,
    help='Number of parallel processing slices (map phase only).',
)
@click.option(
    '--slice-index',
    type=int,
    default=0,
    help='Index of the slice to process (0-based, map phase only).',
)
def trajectories_to_grid(
    input_store: Path,
    mission_db_file: Path | None,
    filter_file: Path | None,
    grid_file: Path,
    mode: str,
    output_times: str,
    output_file: Path | None,
    map_prefix: str,
    slice_count: int,
    slice_index: int,
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d  %(levelname)s/%(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.captureWarnings(True)

    # TODO: Support monthly and (maybe) daily output times.
    if output_times != 'annual':
        raise NotImplementedError(
            f'Output time resolution {output_times} is not supported yet.'
        )

    # Set up filter expression for extracting trajectories from mission
    # database, if provided.
    filter_expr = None
    if filter_file is not None:
        if mission_db_file is None:
            raise click.UsageError(
                'Mission database file must be provided if filter is used.'
            )
        with open(filter_file, 'rb') as fp:
            filter_data = tomllib.load(fp)
            filter_expr = Filter.model_validate(filter_data)

    store = TrajectoryStore.open(base_file=input_store)
    species = list(store[0].trajectory_emissions.keys())
    print(f'Species in trajectory store: {species}')
    grid = Grid.load(grid_file)

    match mode:
        case 'map':
            # Map mode: process trajectories in chunks and save intermediate
            # grid files.
            # TODO: Make trajectory iterator.
            nmissions = _count_missions(store, mission_db_file, filter_expr)
            limit, offset = _slice_limits(nmissions, slice_count, slice_index)
            traj_iter = _trajectory_iterator(
                store, mission_db_file, filter_expr, limit, offset
            )
            logger.info('Flights to process in slice: %s', limit)
            map_output = f'{map_prefix}-{slice_index:05d}.zarr'
            t0 = time.perf_counter()
            map_phase(limit, species, traj_iter, grid, map_output)
            logger.info('map_phase elapsed: %.3f s', time.perf_counter() - t0)

        case 'reduce':
            # Reduce mode: read intermediate grid files and combine into final
            # output.
            if output_file is None:
                raise click.UsageError('Output file must be provided in reduce mode.')
            if mission_db_file is None:
                raise click.UsageError(
                    'Mission database file must be provided in reduce mode.'
                )
            if filter_file is not None:
                raise click.UsageError('--filter-file is not supported in reduce mode.')
            if slice_count != 1:
                raise click.UsageError('--slice-count is not supported in reduce mode.')
            if slice_index != 0:
                raise click.UsageError('--slice-index is not supported in reduce mode.')
            reduce_phase(
                grid,
                species,
                map_prefix,
                output_file,
                mission_db_file,
                input_store,
                grid_file,
            )


def _count_missions(
    store: TrajectoryStore, mission_db_file: Path | None, filter_expr: Filter | None
) -> int:
    if mission_db_file is None:
        # If no mission database or filter is provided, we can just count the
        # number of trajectories in the store.
        return len(store)

    # Otherwise we need to count the number of missions matching the filter
    # conditions in the mission database.
    db = Database(mission_db_file)
    count_query = CountQuery(filter=filter_expr)
    nmissions = db(count_query)
    assert isinstance(nmissions, int)
    return nmissions


def _slice_limits(
    nmissions: int, slice_count: int, slice_index: int
) -> tuple[int, int]:
    # Limit and offset values to use based on slice information. These are used
    # either in the LIMIT and OFFSET clauses in an SQL query or for indexing
    # into the trajectory store. This splits the query results into more or
    # less equally sized groups. The limit for the last slice is adjusted to
    # fit the number of missions.
    limit = math.ceil(nmissions / slice_count)
    offset = limit * slice_index
    if slice_index == slice_count - 1:
        limit = min(limit, nmissions - offset)
    return limit, offset


def _trajectory_iterator(
    store: TrajectoryStore,
    mission_db_file: Path | None,
    filter_expr: Filter | None,
    limit: int,
    offset: int,
) -> Generator[Trajectory]:
    if mission_db_file is None or filter_expr is None:
        # If no mission database is provided or there's no query, we can just
        # iterate through the trajectories in the store. Use iter_range so
        # trajectories are loaded in batched slab reads rather than one at a
        # time.
        yield from store.iter_range(offset, offset + limit)
    else:
        # Otherwise we need to query the mission database for missions matching
        # the filter conditions, and then retrieve the corresponding
        # trajectories from the store. Collect all flight IDs up front so we
        # can do a single bulk index lookup and batched slab reads via
        # iter_flight_ids (instead of per-trajectory get_flight calls).
        with Database(mission_db_file) as db:
            result = db(Query(filter=filter_expr, limit=limit, offset=offset))
            assert isinstance(result, Generator)
            flight_ids = [flight.flight_id for flight in result]
        yield from store.iter_flight_ids(flight_ids)
