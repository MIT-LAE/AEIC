import numpy as np
import pytest

import AEIC.trajectories.builders as tb
from AEIC.missions import Mission
from AEIC.missions.mission import iso_to_timestamp
from AEIC.storage import FieldMetadata, FieldSet
from AEIC.trajectories import TrajectoryStore


@pytest.fixture
def example_mission():
    return Mission(
        origin='BOS',
        destination='LAX',
        departure=iso_to_timestamp('2024-09-01T12:00:00'),
        arrival=iso_to_timestamp('2024-09-01T18:00:00'),
        aircraft_type='738',
        load_factor=1.0,
    )


@pytest.fixture
def example_mission_with_weather():
    return Mission(
        origin='BOS',
        destination='JFK',
        departure=iso_to_timestamp('2024-09-01T12:00:00'),
        arrival=iso_to_timestamp('2024-09-01T13:30:00'),
        aircraft_type='220',
        load_factor=1.0,
    )


@pytest.fixture
def iteration_params():
    return dict(test_reltol=1e-6, test_maxiters=1000)


test_fields = FieldSet(
    'test_fields',
    test_field1=FieldMetadata(
        field_type=np.float32, description='A test field 1', units='unit1'
    ),
    test_field2=FieldMetadata(
        field_type=np.int32, description='A test field 2', units='unit2'
    ),
)


def test_trajectory_simulation_single(sample_missions, performance_model):
    """A simulated trajectory has all three required flight phases
    populated and respects basic mass invariants. Pure `len(traj) > 10`
    would have passed even if `LegacyBuilder` truncated every trajectory
    to 11 points or skipped an entire phase.
    """
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))

    mis = sample_missions[0]
    traj = builder.fly(performance_model, mis)

    assert len(traj) > 10
    assert traj.n_climb > 0
    assert traj.n_cruise > 0
    assert traj.n_descent > 0
    assert traj.n_climb + traj.n_cruise + traj.n_descent <= len(traj)

    # Fuel was burned but nothing else changed: aircraft mass finishes
    # heavier than empty and lighter than the starting mass; fuel mass
    # strictly decreases.
    assert performance_model.empty_mass < float(traj.aircraft_mass[-1])
    assert float(traj.aircraft_mass[-1]) < float(traj.starting_mass)
    assert float(traj.fuel_mass[0]) > float(traj.fuel_mass[-1]) >= 0


@pytest.mark.forked
def test_trajectory_simulation_basic(tmp_path, sample_missions, performance_model):
    fname = tmp_path / 'test_trajectories.nc'

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    ts = TrajectoryStore.create(base_file=fname)

    rng = np.random.default_rng(0)
    original_fields = []
    for mis in sample_missions:
        traj = builder.fly(performance_model, mis)
        traj.add_fields(test_fields)
        traj.test_field1 = rng.random(len(traj), dtype=np.float32)
        traj.test_field2 = rng.integers(0, 100, size=len(traj), dtype=np.int32)
        original_fields.append((traj.test_field1.copy(), traj.test_field2.copy()))
        ts.add(traj)

    ts.close()

    ts_loaded = TrajectoryStore.open(base_file=fname)
    assert len(ts_loaded) == len(ts)
    for traj_loaded, (f1, f2) in zip(ts_loaded, original_fields):
        assert np.array_equal(traj_loaded.test_field1, f1)
        assert np.array_equal(traj_loaded.test_field2, f2)


@pytest.mark.forked
def test_trajectory_simulation_outside_weather_domain(
    example_mission, performance_model
):
    builder = tb.LegacyBuilder(options=tb.Options(use_weather=True, iterate_mass=False))

    with pytest.raises(ValueError):
        builder.fly(performance_model, example_mission)


@pytest.mark.forked
def test_trajectory_simulation_weather(example_mission_with_weather, performance_model):
    """Use-weather builds a trajectory whose ground speed differs from
    the no-weather baseline: without that, `len(traj) > 0` would pass
    even if `use_weather=True` had no effect.
    """
    builder_wx = tb.LegacyBuilder(
        options=tb.Options(use_weather=True, iterate_mass=False)
    )
    builder_nowx = tb.LegacyBuilder(
        options=tb.Options(use_weather=False, iterate_mass=False)
    )

    traj_wx = builder_wx.fly(performance_model, example_mission_with_weather)
    traj_nowx = builder_nowx.fly(performance_model, example_mission_with_weather)

    assert len(traj_wx) > 0
    assert np.any(traj_wx.ground_speed != traj_nowx.ground_speed)


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


@pytest.mark.parametrize(
    'cls',
    [tb.TASOPTBuilder, tb.ADSBBuilder, tb.DymosBuilder],
    ids=['TASOPT', 'ADSB', 'Dymos'],
)
def test_stub_builder_raises_not_implemented(cls):
    """Stub builders must raise on construction so a half-implemented
    subclass can't quietly land — the `NotImplementedError` is the only
    surface saying "this isn't ready" and nothing else polices it.
    """
    with pytest.raises(NotImplementedError):
        cls()


def test_trajectory_performance_model_selector(
    performance_model_selector, sample_missions
):
    """Builder threaded with a selector dispatches each mission via the
    selector. Pure `len(traj) > 0` would have passed even if every
    dispatch silently returned the default model.
    """
    expected_models = [
        'B738',
        'B738',
        'B738',
        'B738',
        'B738',
        'A380',
        'A380',
        'B738',
        'B738',
        'A380',
    ]
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    for mis, expected in zip(sample_missions, expected_models):
        assert performance_model_selector(mis).aircraft_name == expected
        traj = builder.fly(performance_model_selector, mis)
        assert len(traj) > 0
