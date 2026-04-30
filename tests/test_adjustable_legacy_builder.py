import numpy as np
import pytest

import AEIC.trajectories.builders as tb
from AEIC.performance.models.legacy import ROCDFilter
from AEIC.performance.types import AircraftState, SimpleFlightRules
from AEIC.trajectories import GroundTrack
from AEIC.trajectories.builders.adjustable_legacy import AdjustableLegacyContext
from AEIC.units import FL_TO_METERS


def _ground_track(mission):
    return GroundTrack.great_circle(
        mission.origin_position.location,
        mission.destination_position.location,
        allow_overstep=True,
    )


def _expected_starting_mass(
    performance_model,
    mission,
    cruise_altitude,
    reserve_fuel,
    divert_distance,
    hold_time,
):
    perf = performance_model.evaluate(
        AircraftState(altitude=cruise_altitude, aircraft_mass='max'),
        SimpleFlightRules.CRUISE,
    )
    lowest_cruise_altitude = (
        min(performance_model.performance_table(ROCDFilter.ZERO).fl) * FL_TO_METERS
    )
    perf_low = performance_model.evaluate(
        AircraftState(altitude=lowest_cruise_altitude, aircraft_mass='min'),
        SimpleFlightRules.CRUISE,
    )

    approx_time = _ground_track(mission).total_distance / perf.true_airspeed
    fuel_mass = approx_time * perf.fuel_flow
    payload_mass = performance_model.maximum_payload * mission.load_factor
    divert_mass = divert_distance / perf.true_airspeed * perf.fuel_flow
    hold_mass = hold_time * perf_low.fuel_flow

    return min(
        performance_model.empty_mass
        + payload_mass
        + fuel_mass
        + reserve_fuel
        + divert_mass
        + hold_mass,
        performance_model.maximum_mass,
    )


def test_adjustable_legacy_without_adjustments_matches_legacy(
    sample_missions, performance_model
):
    mission = sample_missions[0]
    options = tb.Options(iterate_mass=False)

    legacy_traj = tb.LegacyBuilder(options=options).fly(performance_model, mission)
    adjustable_traj = tb.AdjustableLegacyBuilder(options=options).fly(
        performance_model, mission
    )

    assert adjustable_traj.approx_eq(legacy_traj)
    assert adjustable_traj.starting_mass == pytest.approx(legacy_traj.starting_mass)
    assert adjustable_traj.total_fuel_mass == pytest.approx(legacy_traj.total_fuel_mass)
    assert adjustable_traj.n_climb == legacy_traj.n_climb
    assert adjustable_traj.n_cruise == legacy_traj.n_cruise
    assert adjustable_traj.n_descent == legacy_traj.n_descent


def test_fixed_geometry_adjustments_shape_trajectory(
    sample_missions, performance_model
):
    mission = sample_missions[0]
    descent_distance = 200_000.0

    traj = tb.AdjustableLegacyBuilder(options=tb.Options(iterate_mass=False)).fly(
        performance_model,
        mission,
        climb_start_altitude=1500.0,
        cruise_altitude=9000.0,
        descent_end_altitude=1200.0,
        descent_distance=descent_distance,
    )

    assert traj.altitude[0] == pytest.approx(1500.0)
    assert np.max(traj.altitude) == pytest.approx(9000.0)
    assert traj.altitude[-1] == pytest.approx(1200.0)

    descent_start_idx = np.flatnonzero(np.diff(traj.altitude) < 0)[0]
    expected_descent_start = _ground_track(mission).total_distance - descent_distance
    assert traj.ground_distance[descent_start_idx] == pytest.approx(
        expected_descent_start
    )


def test_fixed_fuel_policy_adjustments_set_starting_mass(
    sample_missions, performance_model
):
    mission = sample_missions[0]
    cruise_altitude = 9000.0
    reserve_fuel = 1234.0
    divert_distance = 150_000.0
    hold_time = 1200.0

    traj = tb.AdjustableLegacyBuilder(options=tb.Options(iterate_mass=False)).fly(
        performance_model,
        mission,
        cruise_altitude=cruise_altitude,
        reserve_fuel=reserve_fuel,
        divert_distance=divert_distance,
        hold_time=hold_time,
    )

    assert traj.starting_mass == pytest.approx(
        _expected_starting_mass(
            performance_model,
            mission,
            cruise_altitude,
            reserve_fuel,
            divert_distance,
            hold_time,
        )
    )


def test_callable_adjustments_receive_context_and_kwargs(
    sample_missions, performance_model
):
    mission = sample_missions[0]
    calls = []

    def adjustment(name, value):
        def _adjust(context, mission_arg, performance_arg, **kwargs):
            calls.append((name, context, mission_arg, performance_arg, kwargs))
            return value

        return _adjust

    reserve_fuel = 1234.0
    divert_distance = 150_000.0
    hold_time = 1200.0
    cruise_altitude = 9000.0

    traj = tb.AdjustableLegacyBuilder(options=tb.Options(iterate_mass=False)).fly(
        performance_model,
        mission,
        climb_start_altitude=adjustment('climb_start_altitude', 1500.0),
        cruise_altitude=adjustment('cruise_altitude', cruise_altitude),
        descent_end_altitude=adjustment('descent_end_altitude', 1200.0),
        descent_distance=adjustment('descent_distance', 200_000.0),
        reserve_fuel=adjustment('reserve_fuel', reserve_fuel),
        divert_distance=adjustment('divert_distance', divert_distance),
        hold_time=adjustment('hold_time', hold_time),
    )

    call_by_name = {name: call for name, *call in calls}
    assert set(call_by_name) == {
        'climb_start_altitude',
        'cruise_altitude',
        'descent_end_altitude',
        'descent_distance',
        'reserve_fuel',
        'divert_distance',
        'hold_time',
    }

    for context, mission_arg, performance_arg, _ in call_by_name.values():
        assert isinstance(context, AdjustableLegacyContext)
        assert mission_arg is mission
        assert performance_arg is performance_model

    assert call_by_name['climb_start_altitude'][3] == {}
    assert call_by_name['cruise_altitude'][3] == {}
    assert call_by_name['descent_end_altitude'][3] == {}
    assert call_by_name['descent_distance'][3] == {}
    assert set(call_by_name['reserve_fuel'][3]) == {'fuel_mass'}
    assert set(call_by_name['divert_distance'][3]) == {'approx_time'}
    assert set(call_by_name['hold_time'][3]) == {'approx_time'}

    assert traj.altitude[0] == pytest.approx(1500.0)
    assert traj.starting_mass == pytest.approx(
        _expected_starting_mass(
            performance_model,
            mission,
            cruise_altitude,
            reserve_fuel,
            divert_distance,
            hold_time,
        )
    )


@pytest.mark.parametrize(
    'kwargs,expected_altitude',
    [
        pytest.param(
            {'climb_start_altitude': 2000.0, 'cruise_altitude': 1000.0},
            2000.0,
            id='cruise_below_climb_clamps_to_climb',
        ),
        pytest.param(
            {'cruise_altitude': 20_000.0},
            None,
            id='cruise_above_ceiling_clamps_to_ceiling',
        ),
    ],
)
def test_adjustment_altitude_clamps(
    sample_missions, performance_model, kwargs, expected_altitude
):
    mission = sample_missions[0]
    if expected_altitude is None:
        expected_altitude = performance_model.maximum_altitude

    traj = tb.AdjustableLegacyBuilder(options=tb.Options(iterate_mass=False)).fly(
        performance_model, mission, **kwargs
    )

    assert np.max(traj.altitude) == pytest.approx(expected_altitude)


@pytest.mark.parametrize(
    'kwargs,match',
    [
        pytest.param(
            {'cruise_altitude': 9000.0, 'descent_end_altitude': 10_000.0},
            'Arrival airport \\+ 3000ft',
            id='descent_end_above_descent_start_raises',
        ),
        pytest.param(
            {'descent_distance': -1.0},
            'Descent distance must be non-negative',
            id='negative_descent_distance_raises',
        ),
    ],
)
def test_invalid_adjustments_raise_value_error(
    sample_missions, performance_model, kwargs, match
):
    with pytest.raises(ValueError, match=match):
        tb.AdjustableLegacyBuilder(options=tb.Options(iterate_mass=False)).fly(
            performance_model, sample_missions[0], **kwargs
        )
