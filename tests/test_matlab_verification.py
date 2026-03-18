import tomllib

import pytest

import AEIC.trajectories.builders as tb
from AEIC.emissions import compute_emissions
from AEIC.missions import Mission
from AEIC.performance.models import PerformanceModel
from AEIC.types import Fuel
from AEIC.verification.legacy import LegacyTrajectory
from AEIC.verification.metrics import out_of_tolerance

TRAJ_FIELDS = [
    'flight_time',
    'ground_distance',
    'latitude',
    'longitude',
    'altitude',
    'fuel_flow',
    'aircraft_mass',
    'azimuth',
    'true_airspeed',
    'rate_of_climb',
]

TRAJ_FIELD_UNITS = {
    'flight_time': 's',
    'ground_distance': 'm',
    'altitude': 'm',
    'fuel_flow': 'kg s⁻¹',
    'aircraft_mass': 'kg',
    'true_airspeed': 'm s⁻¹',
    'rate_of_climb': 'm s⁻¹',
}

COMPARISON_FIELDS = TRAJ_FIELDS + ['trajectory_indices']

SKIP_FINAL_POINT_FIELDS = set(['true_airspeed'])


@pytest.mark.config_updates(use_weather=False)
def test_matlab_verification(test_data_dir) -> None:
    # Set up paths to test data.
    data_dir = test_data_dir / 'verification/legacy'
    legacy_dir = data_dir / 'matlab-output'
    missions_file = data_dir / 'missions.toml'
    fuel_file = data_dir / 'fuel.toml'
    perf_path = data_dir / 'performance-model.toml'

    # Load test data: performance model, missions, fuel file.
    pm = PerformanceModel.load(perf_path)
    with open(missions_file, 'rb') as fp:
        mission_dict = tomllib.load(fp)
    missions = Mission.from_toml(mission_dict)
    with open(fuel_file, 'rb') as fp:
        fuel = Fuel.model_validate(tomllib.load(fp))

    # Create a single trajectory builder to fly all missions.
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))

    failed = []

    for mission in missions:
        # Load legacy data for mission and convert to "new AEIC" format. (This
        # includes unit conversions.)
        legacy_traj_in = LegacyTrajectory(legacy_dir / f'{mission.label}.csv')
        legacy_traj = legacy_traj_in.trajectory()

        # Simulate mission and compute emissions with new AEIC.
        new_traj = builder.fly(pm, mission)
        new_traj.add_fields(compute_emissions(pm, fuel, new_traj))

        # For comparison, we do *not* interpolate the new AEIC trajectory
        # onto the same time points as the legacy trajectory. The match
        # should be close enough that we can compare corresponding points
        # along the trajectories. The number of points in the trajectories
        # should match exactly.

        # Compute comparison metrics.
        # dict[str, ComparisonMetrics | SpeciesValues[ComparisonMetrics]]
        metrics = legacy_traj.compare(
            new_traj, COMPARISON_FIELDS, SKIP_FINAL_POINT_FIELDS
        )

        # Record any metrics that are outside tolerance.
        bad_metrics = out_of_tolerance(metrics, mape_pct_tol=0.25)
        if len(bad_metrics) > 0:
            failed.append((mission.label, bad_metrics))

    if len(failed) > 0:
        print('Missions with metrics outside tolerance:')
        for mission_id, bad_metrics in failed:
            print(f'  {mission_id}:')
            for m in bad_metrics:
                print(f'    {m}')

    assert len(failed) == 0, 'Missions with metrics outside tolerance'
