"""Tests for the conversion from current mass to the initial mass of the
PIANO climb and descent tables."""

import re

import pytest

from AEIC.performance.models.base import PerformanceTableInput
from AEIC.performance.models.piano import PianoClimbDescentMassAxis
from AEIC.performance.types import AircraftState
from AEIC.storage import FlightPhase
from AEIC.units import METERS_TO_FL

COLS = ['fl', 'mass', 'tas', 'rocd', 'fuel_flow', 'burn']

# Each mass is one climb that starts at that mass and burns fuel as it goes
# up, so a row holds the values at the current mass `mass - burn`. The heavier
# climb stops at a lower FL. Only `fl`, `mass` and `burn` matter here.
ROWS = [
    [0.0, 60000.0, 100.0, 20.0, 2.0, 0.0],
    [100.0, 60000.0, 150.0, 15.0, 1.5, 500.0],
    [200.0, 60000.0, 200.0, 10.0, 1.0, 1000.0],
    [0.0, 80000.0, 110.0, 16.0, 2.4, 0.0],
    [100.0, 80000.0, 160.0, 12.0, 1.8, 800.0],
    [150.0, 80000.0, 170.0, 8.0, 1.6, 1200.0],
]


def mass_axis(rows=ROWS) -> PianoClimbDescentMassAxis:
    return PianoClimbDescentMassAxis.from_input(
        PerformanceTableInput(cols=COLS, data=rows), FlightPhase.CLIMB
    )


def raises_message(message: str):
    return pytest.raises(ValueError, match=f'^{re.escape(message)}$')


def test_current_mass_of_every_row_gives_its_initial_mass():
    axis = mass_axis()
    for fl, mass, *_, burn in ROWS:
        assert axis.get_corresponding_initial_mass(mass - burn, fl) == mass


@pytest.mark.parametrize(
    ('fl', 'mass'),
    [
        # At FL100 the climbs from 60000 and 80000 kg have current masses of
        # 59500 and 79200 kg. 59500 + 0.25 * (79200 - 59500) = 64425 kg.
        pytest.param(100.0, 64425.0, id='grid-fl'),
        # At FL50 the climbs burned 250 and 400 kg, so the current masses are
        # 59750 and 79600 kg. 59750 + 0.25 * (79600 - 59750) = 64712.5 kg.
        pytest.param(50.0, 64712.5, id='between-fls'),
    ],
)
def test_current_mass_between_initial_masses_interpolates_linearly(fl, mass):
    """The current mass is 0.25 of the way between the current masses of the
    two climbs, so the initial mass is 0.25 of the way from 60000 to 80000 kg.
    The weight is not 0.5, because at 0.5 a swap of the two interpolation
    weights gives the same result."""
    assert mass_axis().get_corresponding_initial_mass(mass, fl) == pytest.approx(
        65000.0, rel=1e-12
    )


def test_current_mass_below_the_lightest_climb_lists_the_current_masses():
    # At FL100 the current masses are 59500 and 79200 kg.
    with raises_message(
        'climb table: current mass 59400.0 at fl=100.0 is not in range of the '
        'tabulated data. Tabulated masses available at this fl: 59500.0 (from '
        'initial mass = 60000.0), 79200.0 (from initial mass = 80000.0)'
    ):
        mass_axis().get_corresponding_initial_mass(59400.0, 100.0)


def test_current_mass_above_every_ceiling_raises():
    with raises_message('climb table: no tabulated initial masses reaches fl=250.0'):
        mass_axis().get_corresponding_initial_mass(59000.0, 250.0)


def test_current_mass_above_the_heavy_ceiling_uses_the_light_climb_alone():
    # The 80000 kg climb stops at FL150. At FL180, 0.8 of the way from FL100
    # to FL200, the 60000 kg climb burned 900 kg.
    axis = mass_axis()

    assert axis.get_corresponding_initial_mass(59100.0, 180.0) == 60000.0
    with raises_message(
        'climb table: current mass 70000.0 at fl=180.0 is not in range of the '
        'tabulated data. Tabulated masses available at this fl: 59100.0 (from '
        'initial mass = 60000.0), none (from initial mass = 80000.0)'
    ):
        axis.get_corresponding_initial_mass(70000.0, 180.0)


@pytest.mark.parametrize('mass', ['min', 'max'])
def test_find_corresponding_initial_state_leaves_min_and_max_mass_unchanged(mass):
    state = AircraftState(altitude=75.0 / METERS_TO_FL, aircraft_mass=mass)
    assert mass_axis().find_corresponding_initial_state(state) == state


def test_find_corresponding_initial_state_converts_the_altitude_to_a_flight_level():
    """The state gives the altitude in metres. The conversion must look up the
    current masses at FL100, which are 59500 and 79200 kg. At 3048 m
    none of the climbs have data."""
    state = AircraftState(altitude=100.0 / METERS_TO_FL, aircraft_mass=64425.0)

    converted = mass_axis().find_corresponding_initial_state(state)

    assert converted.aircraft_mass == pytest.approx(65000.0, rel=1e-12)


def test_heavier_climb_lighter_than_its_neighbour_after_burn_raises():
    rows = [
        [0.0, 60000.0, 100.0, 20.0, 2.0, 0.0],
        [100.0, 60000.0, 150.0, 15.0, 1.5, 500.0],
        [0.0, 61000.0, 110.0, 16.0, 2.4, 0.0],
        [100.0, 61000.0, 160.0, 12.0, 1.8, 2000.0],
    ]
    with raises_message(
        'climb table: at fl=100.0, the climb from initial mass 61000.0 has a '
        'current mass of 59000.0, which is not above the 59500.0 of the climb '
        'from initial mass 60000.0'
    ):
        mass_axis(rows)
