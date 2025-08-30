import multiprocessing
import time

import ray

from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.legacy_trajectory import LegacyTrajectory
from emissions import emission


# Function to load performance model using config file
def load_performance_model(config_file_loc: str):
    perf = PerformanceModel(config_file_loc)
    return perf


# Function that is to be parallelized
@ray.remote
def trajectory_and_emissions(
    performance_model, mission, optimize_traj: bool = False, iterate_mass: bool = False
):
    traj = LegacyTrajectory(performance_model, mission, optimize_traj, iterate_mass)
    traj.fly_flight()
    emissions = emission.Emission(performance_model, traj, True)
    return traj, emissions


# Same function as above without Ray remote feature
def trajectory_and_emissions_serial(
    performance_model, mission, optimize_traj: bool = False, iterate_mass: bool = False
):
    traj = LegacyTrajectory(performance_model, mission, optimize_traj, iterate_mass)
    traj.fly_flight()
    emissions = emission.Emission(performance_model, traj, True)
    return traj, emissions


if __name__ == "__main__":
    config_file_loc = 'IO/default_config.toml'
    perfModel = load_performance_model(config_file_loc)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Parallel Processing
    start = time.time()
    futures = [
        trajectory_and_emissions.remote(perfModel, mission)
        for mission in perfModel.missions
    ]
    results = ray.get(futures)
    end = time.time()
    ray.shutdown()

    print("Parallel Results:", results)
    print(f"Parallel Time Taken: {end - start:.2f} seconds\n")

    time_taken_by_ray = end - start

    # Same example done serially
    serial_start = time.time()
    for mission in perfModel.missions:
        traj, ems = trajectory_and_emissions_serial(perfModel, mission)
    serial_end = time.time()
    time_taken_serial = serial_end - serial_start

    print("🔍 Diagnostics:")
    print(f"Speed: {time_taken_by_ray:.2f}s with Ray")
    print(f"Speed: {time_taken_serial:.2f}s with Serial")
    print(f"Speedup: {time_taken_serial / time_taken_by_ray:.2f}x faster with Ray")
    eff = (time_taken_serial / time_taken_by_ray) / multiprocessing.cpu_count() * 100
    print(f"Efficiency: {eff:.2f}% per core\n")
