import tomllib

import numpy as np
import pytest

import AEIC.trajectories.builders as tb
from AEIC.missions import Mission
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories import FieldMetadata, FieldSet, TrajectoryStore
from AEIC.utils.files import file_location
from AEIC.utils.helpers import iso_to_timestamp


@pytest.fixture(scope='session')
def sample_missions():
    missions_file = file_location('missions/sample_missions_10.toml')
    with open(missions_file, 'rb') as f:
        mission_dict = tomllib.load(f)
    return Mission.from_toml(mission_dict)


@pytest.fixture(scope='session')
def example_mission():
    return Mission(
        origin='BOS',
        destination='LAX',
        departure=iso_to_timestamp('2024-09-01T12:00:00'),
        arrival=iso_to_timestamp('2024-09-01T18:00:00'),
        aircraft_type='738',
        load_factor=1.0,
    )


@pytest.fixture(scope='session')
def example_mission_with_weather():
    return Mission(
        origin='BOS',
        destination='JFK',
        departure=iso_to_timestamp('2024-09-01T12:00:00'),
        arrival=iso_to_timestamp('2024-09-01T13:30:00'),
        aircraft_type='220',
        load_factor=1.0,
    )


@pytest.fixture(scope='session')
def iteration_params():
    return dict(test_reltol=1e-6, test_maxiters=1000)


@pytest.fixture(scope='session')
def performance_model(test_data_dir):
    perf = PerformanceModel(file_location('IO/default_config.toml'))
    perf.config.weather_data_dir = test_data_dir / 'weather'
    return perf


test_fields = FieldSet(
    'test_fields',
    test_field1=FieldMetadata(
        field_type=np.float32, description='A test field 1', units='unit1'
    ),
    test_field2=FieldMetadata(
        field_type=np.int32, description='A test field 2', units='unit2'
    ),
)


def test_trajectory_simulation_basic(tmp_path, sample_missions, performance_model):
    fname = tmp_path / 'test_trajectories.nc'

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    ts = TrajectoryStore.create(base_file=fname)

    for mis in sample_missions:
        traj = builder.fly(performance_model, mis)
        traj.add_fields(test_fields)
        traj.test_field1 = np.random.rand(len(traj))
        traj.test_field2 = np.random.randint(0, 100, size=len(traj))
        ts.add(traj)

    ts.close()

    ts_loaded = TrajectoryStore.open(base_file=fname)
    assert len(ts_loaded) == len(ts)
    # TODO: Test that additional fields are correctly saved and loaded.


def test_trajectory_simulation_outside_weather_domain(
    example_mission, performance_model
):
    builder = tb.LegacyBuilder(options=tb.Options(use_weather=True, iterate_mass=False))

    with pytest.raises(ValueError):
        builder.fly(performance_model, example_mission)


def test_trajectory_simulation_weather(example_mission_with_weather, performance_model):
    builder = tb.LegacyBuilder(options=tb.Options(use_weather=True, iterate_mass=False))

    traj = builder.fly(performance_model, example_mission_with_weather)

    assert len(traj) > 0


def test_trajectory_mass_iter(performance_model, example_mission, iteration_params):
    """Test that:
    - Mass iteration is consistent between start and end of mission
    - The final mass residual is less than required for convergence
    """

    builder_wout_iter = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))

    builder_with_iter = tb.LegacyBuilder(
        options=tb.Options(
            iterate_mass=True,
            max_mass_iters=iteration_params['test_maxiters'],
            mass_iter_reltol=iteration_params['test_reltol'],
        )
    )

    traj_wout_iter = builder_wout_iter.fly(performance_model, example_mission)
    traj_with_iter = builder_with_iter.fly(performance_model, example_mission)

    fuel_difference = traj_wout_iter.fuel_mass[0] - traj_with_iter.fuel_mass[0]
    mass_difference = traj_wout_iter.aircraft_mass[0] - traj_with_iter.aircraft_mass[0]
    final_mass_residual = traj_with_iter.fuel_mass[-1] / traj_with_iter.fuel_mass[0]

    assert mass_difference == pytest.approx(fuel_difference)
    assert final_mass_residual < iteration_params['test_reltol']


def test_trajectory_mass_iter_fail(
    performance_model, example_mission, iteration_params
):
    """Test that in case without enough iterations to meet reltol, error is raised"""

    builder_fail = tb.LegacyBuilder(
        options=tb.Options(
            iterate_mass=True,
            max_mass_iters=1,
            mass_iter_reltol=iteration_params['test_reltol'],
        )
    )

    with pytest.raises(RuntimeError):
        builder_fail.fly(performance_model, example_mission)
