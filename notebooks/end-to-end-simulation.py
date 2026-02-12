# The idea here is (steps 1 and 2 in this script, step 3 in a notebook):
#
# 1. Demonstrate running trajectory simulations based on missions taken from
#    OAG data that has ingested into a missions database;
# 2. Save the simulation results to NetCDF files;
# 3. Perform subset queries on the mission database and retrieve the
#    corresponding trajectories to calculate some (not very realistic)
#    statistics.
#
# The use case is silly and the mission database is the small one used for the
# test suite, but it should hopefully serve as some sort of inspiration!

import multiprocessing as mp
import os
import shutil
import tomllib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import AEIC.trajectories.builders as tb
from AEIC.config import Config, config
from AEIC.emissions import compute_emissions
from AEIC.missions import CountQuery, Database, Mission, Query
from AEIC.performance.models import PerformanceModel
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.types import Fuel

# ------------------------------------------------------------------------------
#
#  SETUP

# Set the AEIC path to include the main data directory and the test data
# directory (which includes the mission database) and load the default AEIC
# configuration.

os.environ['AEIC_PATH'] = str(Path.cwd().parent / 'tests' / 'data')
Config.load()


# Load single performance model to use for all simulations.

perf = PerformanceModel.load(
    config.file_location('performance/sample_performance_model.toml')
)


# Load fuel data.

with open(config.emissions.fuel_file, 'rb') as fp:
    fuel = Fuel.model_validate(tomllib.load(fp))


# Function to set up legacy trajectory builders to simulate flights. We use a
# function here so that we can create these things on demand in the worker
# function below. We disable the mass iteration because it causes lots of
# exceptions due to non-convergence. (I think they just used to be warnings,
# but they were happening so often that I think they need some attention, so I
# made them exceptions.)


def builder_maker():
    return tb.LegacyBuilder(options=tb.Options(iterate_mass=False))


# ------------------------------------------------------------------------------
#
#  TRAJECTORY SIMULATION CODE

# The code here simulates all missions in a mission database using a single
# performance model and saves the results to a trajectory store. The
# trajectories are simulated in parallel using a process pool.
#
# Note how we use `with` everywhere for both the trajectory store and the
# mission database to ensure prompt cleanup!
#
# First, the outer driver function:


def simulate_all(
    mission_db_file: str,
    output_store: str,
    builder_maker: Callable[[], tb.Builder],
    n_jobs: int = 8,
):
    # Count flights to be simulated. We do this once to decide how many
    # flights to allocate to each worker process.
    with Database(mission_db_file) as db:
        nflights = db(CountQuery())
    assert isinstance(nflights, int)
    print(f'Flights to process: {nflights}')

    # Limit and offset values to use in each worker job. These are used
    # directly in the LIMIT and OFFSET clauses in an SQL query. This splits the
    # query results into more or less equally sized groups.
    limits = [round(nflights / n_jobs)] * (n_jobs - 1)
    limits.append(nflights - sum(limits))
    offsets = [round(nflights / n_jobs) * job for job in range(n_jobs)]

    # Run slices of the database query result set in a process worker pool. The
    # process pool is created using the "spawn" context to avoid issues with
    # the NetCDF dependencies used for writing trajectory stores.
    print('Running slices of size:', ', '.join([str(lim) for lim in limits]))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
        # This pattern, of making a dictionary whose keys are futures running
        # the process pool and whose values are some identifier for each slice,
        # is a common one when using concurrent.futures.
        futures = {
            pool.submit(
                simulate_slice, idx, limit, offset, mission_db_file, builder_maker
            ): idx
            for idx, limit, offset in zip(range(1, n_jobs + 1), limits, offsets)
        }

        # Retrieve diagnostic results from futures as they complete and print
        # some messages.
        total_failed = 0
        slice_files = []
        print('')
        for future in as_completed(futures):
            slice_idx = futures[future]
            slice_file, nfailed, exceptions = future.result()
            total_failed += nfailed
            slice_files.append(slice_file)
            print(f'SLICE {slice_idx} COMPLETED ({nfailed} SIMULATIONS FAILED)')
            for exc in exceptions:
                print(exc)
        print('')
        print(f'Total failed simulations: {total_failed}')

    # Merge NetCDF files into single trajectory store.
    print('')
    print(f'Merging stores => {output_store}')

    # Deferred import so that the netCDF4 package doesn't get imported earlier
    # in the main process (can cause crashes).
    from AEIC.trajectories import TrajectoryStore

    # Merge trajectory stores from all slices.
    TrajectoryStore.merge(output_store, slice_files)


# And now the function that runs the simulations for a single slice:


def simulate_slice(
    slice_idx: int,
    limit: int,
    offset: int,
    mission_db_file: str,
    builder_maker: Callable[[], tb.Builder],
):
    # Deferred import to keep state of netCDF4 package "clean" in the main
    # process.
    from AEIC.trajectories import TrajectoryStore

    # There is one output file per slice. We'll merge them into a merged
    # trajectory store when we're done.
    output_file = f'slice-{slice_idx}.nc'
    if os.path.exists(output_file):
        os.remove(output_file)

    with TrajectoryStore.create(base_file=output_file) as ts:
        with Database(mission_db_file) as db:
            # Record failure information for return value.
            nfailed = 0
            exceptions = []

            # Report exceptions along with some mission information.
            def report(exc, mis):
                nonlocal exceptions
                exceptions.append(
                    f'{exc}: {mis.origin} -> {mis.destination} = '
                    f'{mis.gc_distance / 1000:0.2f} km'
                )

            # For short missions, the legacy trajectory builder tries to step
            # beyond the great circle distance between the origin and
            # destination points. We keep track of (origin, destination) pairs
            # here to avoid reporting repeated problems like this.
            short: set[tuple[str, str]] = set()

            # Create a trajectory builder.
            builder = builder_maker()

            # Retrieve all flights in this slice and simulate them one by one.
            for flight in db(Query(limit=limit, offset=offset)):  # type: ignore
                # Create a mission from the flight database result. (We fix
                # the load factor here.)
                mission = Mission.from_query_result(flight, load_factor=1.0)

                # Fly the mission, catching exceptions.
                try:
                    traj = builder.fly(perf, mission)
                    traj.add_fields(compute_emissions(perf, fuel, traj))
                except ValueError as exc:
                    # General exception from trajectory builder.
                    report(exc, mission)
                    nfailed += 1
                    continue
                except GroundTrack.Exception as exc:
                    # Exception from ground track interpolator: usually
                    # because the builder has tried to step beyond the end of
                    # the great circle track between the origin and destination
                    # (this happens for very short flights).
                    if (mission.origin, mission.destination) not in short:
                        short.add((mission.origin, mission.destination))
                        report(exc, mission)
                    nfailed += 1
                    continue

                # Save the trajectory.
                ts.add(traj)

            # Return information passed back through the process pool future.
            return output_file, nfailed, exceptions


# ------------------------------------------------------------------------------
#
#  MAIN FUNCTION
#


def run():
    # Select mission database (we just use a small test database here).
    mission_db_file = config.file_location('missions/oag-2019-test-subset.sqlite')

    # Clean up any existing output store.
    output_store = 'test-simulations.aeic-store'
    shutil.rmtree(output_store, ignore_errors=True)

    # Run all simulations from the mission database.
    simulate_all(mission_db_file, output_store, builder_maker)


if __name__ == '__main__':
    run()
