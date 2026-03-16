import logging
import math
import tomllib
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import AEIC.trajectories.builders as tb
from AEIC.config import Config, config
from AEIC.emissions import compute_emissions
from AEIC.missions import CountQuery, Database, Mission, Query
from AEIC.performance.model_selector import (
    PerformanceModelSelector,
    SimplePerformanceModelSelector,
)
from AEIC.performance.models import BasePerformanceModel, PerformanceModel
from AEIC.trajectories import TrajectoryStore
from AEIC.types import Fuel

logger = logging.getLogger(__name__)


def simulate_slice(
    slice_idx: int,
    limit: int,
    offset: int,
    output_store: Path,
    mission_db_file: Path,
    performance_model: BasePerformanceModel | PerformanceModelSelector,
    fuel: Fuel,
    builder: tb.Builder,
):

    # There is one output file per slice. We'll merge them into a merged
    # trajectory store when we're done.
    output_file = Path(f'{str(output_store)}-{slice_idx:03d}.nc')
    if output_file.exists():
        output_file.unlink()

    with TrajectoryStore.create(base_file=output_file) as ts:
        with Database(mission_db_file) as db:
            # Record failure information for final report.
            nfailed = 0

            # Retrieve all flights in this slice and simulate them one by one.
            with logging_redirect_tqdm():
                for flight in tqdm(db(Query(limit=limit, offset=offset)), total=limit):  # type: ignore
                    # Create a mission from the flight database result. (We fix
                    # the load factor here.)
                    mission = Mission.from_query_result(flight, load_factor=1.0)

                    # Fly the mission, catching exceptions.
                    try:
                        pm = performance_model
                        if isinstance(performance_model, PerformanceModelSelector):
                            pm = performance_model(mission)
                        assert isinstance(pm, BasePerformanceModel)
                        traj = builder.fly(pm, mission)
                        traj.add_fields(compute_emissions(pm, fuel, traj))
                    except Exception:
                        # General exception from trajectory builder.
                        logger.exception(
                            'Error simulating mission '
                            f'{mission.origin} -> {mission.destination} '
                            f'({mission.gc_distance / 1000:0.2f} km):'
                        )
                        nfailed += 1
                        continue

                    # Save the trajectory.
                    ts.add(traj)

            # Return information passed back through the process pool future.
            logger.info(f'Slice {slice_idx} complete: {nfailed} failed simulations.')


@click.command()
@click.option(
    '--config-file',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Configuration file path.',
)
@click.option(
    '--performance-selector-dir',
    type=click.Path(exists=True, path_type=Path),
    help='Performance model selector path.',
)
@click.option(
    '--performance-model-file',
    type=click.Path(exists=True, path_type=Path),
    help='Performance model path.',
)
@click.option(
    '--mission-db-file',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Mission database path.',
)
@click.option(
    '--output-store',
    type=click.Path(path_type=Path),
    required=True,
    help='Output trajectory store path prefix.',
)
@click.option(
    '--slice-count', type=int, default=1, help='Number of parallel processing slices.'
)
@click.option(
    '--slice-index',
    type=int,
    default=0,
    help='Index of the slice to process (0-based).',
)
def run_simulations(
    config_file: Path,
    performance_selector_dir: Path | None,
    performance_model_file: Path | None,
    mission_db_file: Path,
    output_store: Path,
    slice_count: int,
    slice_index: int,
):
    if performance_selector_dir is None == performance_model_file is None:
        raise click.UsageError(
            'Exactly one of --performance-selector-dir or '
            '--performance-model-file must be provided.'
        )
    logging.basicConfig(level=logging.INFO)

    # Load given configuration file.
    Config.load(config_file)

    # Count flights to be simulated. We do this once to decide how many
    # flights to allocate to each worker process.
    with Database(mission_db_file) as db:
        nflights = db(CountQuery())
    assert isinstance(nflights, int)
    logger.info('Total flights to process: %s', nflights)

    # Limit and offset values to use based on slice information. These are used
    # directly in the LIMIT and OFFSET clauses in an SQL query. This splits the
    # query results into more or less equally sized groups. The limit for the
    # last slice is adjusted to fit the number of flights.
    limit = math.ceil(nflights / slice_count)
    offset = limit * slice_index
    if slice_index == slice_count - 1:
        limit = min(limit, nflights - offset)
    logger.info('Flights to process in slice: %s', limit)

    # Load single performance model to use for all simulations.
    if performance_selector_dir is not None:
        performance_model = SimplePerformanceModelSelector(performance_selector_dir)
    else:
        assert performance_model_file is not None
        performance_model = PerformanceModel.load(performance_model_file)

    # Load fuel data.
    with open(config.emissions.fuel_file, 'rb') as fp:
        fuel = Fuel.model_validate(tomllib.load(fp))

    # Make trajectory builder.
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))

    simulate_slice(
        slice_index,
        limit,
        offset,
        output_store,
        mission_db_file,
        performance_model,
        fuel,
        builder,
    )
