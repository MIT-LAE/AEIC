import logging
import math
import tomllib
from collections.abc import Generator
from pathlib import Path

import click
import zarr
from tqdm import tqdm

from AEIC.gridding.grid import Grid
from AEIC.missions import CountQuery, Database, Filter, Query
from AEIC.trajectories import Trajectory, TrajectoryStore

logger = logging.getLogger(__name__)


def map_phase(
    ntrajs: int,
    traj_iter: Generator[Trajectory],
    grid,
    map_output: str,
):
    output = None
    for traj in tqdm(traj_iter, total=ntrajs):
        if output is None:
            nspecies = len(traj.trajectory_emissions)
            shape = grid.shape + (nspecies,)
            output = zarr.create_array(store=map_output, dtype='f4', shape=shape)

        lat_idx, lon_idx, alt_idx = grid.get_cell_indices(traj)

        pass


def reduce_phase(
    grid,
    map_prefix: str,
    output_file: Path,
):
    pass


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
            map_output = f'{map_prefix}-{slice_index:03d}.zarr'
            map_phase(nmissions, traj_iter, grid, map_output)

        case 'reduce':
            # Reduce mode: read intermediate grid files and combine into final
            # output.
            if output_file is None:
                raise click.UsageError('Output file must be provided in reduce mode.')
            reduce_phase(grid, map_prefix, output_file)


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
    if mission_db_file is None:
        # If no mission database is provided, we can just iterate through the
        # trajectories in the store.
        for idx in range(offset, offset + limit):
            yield store[idx]
    else:
        # Otherwise we need to query the mission database for missions matching
        # the filter conditions, and then retrieve the corresponding
        # trajectories from the store.
        with Database(mission_db_file) as db:
            result = db(Query(filter=filter_expr, limit=limit, offset=offset))
            assert isinstance(result, Generator)
            for flight in result:
                traj = store.get_flight(flight.flight_id)
                if traj is not None:
                    yield traj
