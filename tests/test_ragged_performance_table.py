"""Tests for `RaggedPerformanceTable`, on synthetic tables."""

import math
import re

import pandas as pd
import pytest

from AEIC.performance.models.base import PerformanceTableInput, ROCDFilter
from AEIC.performance.models.ragged import RaggedPerformanceTable
from AEIC.performance.types import AircraftState, Performance
from AEIC.units import METERS_TO_FL

CLIMB_AXES = ('mass', 'fl')
CRUISE_AXES = ('fl', 'mass', 'mach')

CLIMB_COLS = ['fl', 'tas', 'rocd', 'fuel_flow', 'mass']

# Two masses with different FL points. The heavier mass stops at a lower FL.
CLIMB_ROWS = [
    [0.0, 100.0, 20.0, 2.0, 60000.0],
    [100.0, 150.0, 15.0, 1.5, 60000.0],
    [200.0, 200.0, 10.0, 1.0, 60000.0],
    [0.0, 110.0, 16.0, 2.4, 80000.0],
    [150.0, 170.0, 8.0, 1.6, 80000.0],
]

CRUISE_COLS = ['fl', 'mass', 'mach', 'tas', 'rocd', 'fuel_flow']

CRUISE_ROWS = [
    [300.0, 60000.0, 0.70, 210.0, 0.0, 1.0],
    [300.0, 60000.0, 0.80, 240.0, 0.0, 1.4],
    [300.0, 80000.0, 0.70, 211.0, 0.0, 1.2],
    [300.0, 80000.0, 0.80, 241.0, 0.0, 1.6],
    [350.0, 60000.0, 0.70, 205.0, 0.0, 0.9],
    [350.0, 60000.0, 0.80, 235.0, 0.0, 1.3],
    [350.0, 80000.0, 0.75, 220.0, 0.0, 1.3],
    [350.0, 80000.0, 0.80, 236.0, 0.0, 1.5],
]

DESCENT_ROWS = [
    [0.0, 120.0, -10.0, 0.3, 70000.0],
    [200.0, 180.0, -12.0, 0.2, 70000.0],
]


def climb_table(rows=CLIMB_ROWS, cols=CLIMB_COLS) -> RaggedPerformanceTable:
    ptin = PerformanceTableInput(cols=cols, data=rows)
    return RaggedPerformanceTable.from_input(ptin, ROCDFilter.POSITIVE, CLIMB_AXES)


def cruise_table(rows=CRUISE_ROWS) -> RaggedPerformanceTable:
    ptin = PerformanceTableInput(cols=CRUISE_COLS, data=rows)
    return RaggedPerformanceTable.from_input(ptin, ROCDFilter.ZERO, CRUISE_AXES)


def raises_message(message: str):
    return pytest.raises(ValueError, match=f'^{re.escape(message)}$')


def state(fl: float, mass: float | str) -> AircraftState:
    return AircraftState(altitude=fl / METERS_TO_FL, aircraft_mass=mass)  # type: ignore[arg-type]


def assert_performance(perf: Performance, tas: float, rocd: float, ff: float):
    assert perf.true_airspeed == pytest.approx(tas, rel=1e-12)
    assert perf.rate_of_climb == pytest.approx(rocd, rel=1e-12)
    assert perf.fuel_flow == pytest.approx(ff, rel=1e-12)


def test_every_climb_row_returns_its_tabulated_values():
    table = climb_table()
    for fl, tas, rocd, ff, mass in CLIMB_ROWS:
        assert_performance(table.interpolate(state(fl, mass)), tas, rocd, ff)


def test_every_cruise_row_returns_its_tabulated_values():
    table = cruise_table()
    for fl, mass, mach, tas, rocd, ff in CRUISE_ROWS:
        assert_performance(table.interpolate(state(fl, mass), mach=mach), tas, rocd, ff)


def test_altitude_in_metres_maps_to_flight_level():
    table = climb_table()

    # FL 50 is halfway between FL 0 and FL 100 on the lighter mass.
    altitude = 50.0 / METERS_TO_FL
    perf = table.interpolate(AircraftState(altitude=altitude, aircraft_mass=60000.0))
    assert_performance(perf, 125.0, 17.5, 1.75)


def test_row_order_does_not_change_results():
    forward = climb_table()
    backward = climb_table(rows=CLIMB_ROWS[::-1])
    query = state(75.0, 65000.0)

    assert forward.interpolate(query) == backward.interpolate(query)


def test_extra_data_columns_are_ignored():
    rows = [[*row, 999.0] for row in CLIMB_ROWS]
    table = climb_table(rows=rows)

    assert list(table.df.columns) == CLIMB_COLS
    assert table.interpolate(state(0.0, 60000.0)) == Performance(100.0, 20.0, 2.0)


def test_extra_outputs_are_interpolated_by_name():
    # Add an extra column to the row
    rows = [[*row, 10.0 * i] for i, row in enumerate(CLIMB_ROWS)]
    ptin = PerformanceTableInput(cols=[*CLIMB_COLS, 'extra'], data=rows)
    table = RaggedPerformanceTable.from_input(
        ptin, ROCDFilter.POSITIVE, CLIMB_AXES, extra_outputs=('extra',)
    )

    # FL 50 is halfway between FL 0 and FL 100 on the lighter mass.
    assert table.interpolate_values(state(50.0, 60000.0)) == pytest.approx(
        {'tas': 125.0, 'rocd': 17.5, 'fuel_flow': 1.75, 'extra': 5.0}, rel=1e-12
    )


@pytest.mark.parametrize(
    ('missing', 'extra_outputs'),
    [
        pytest.param('mass', (), id='axis'),
        pytest.param('tas', (), id='output'),
        pytest.param('extra', ('extra',), id='extra_output'),
    ],
)
def test_missing_column_raises(missing, extra_outputs):
    df = pd.DataFrame(CLIMB_ROWS, columns=CLIMB_COLS).drop(
        columns=missing, errors='ignore'
    )
    with raises_message(f'Missing "{missing}" column in climb performance table'):
        RaggedPerformanceTable(
            df, CLIMB_AXES, ROCDFilter.POSITIVE, extra_outputs=extra_outputs
        )


@pytest.mark.parametrize(
    ('mass', 'expected'),
    [('min', (125.0, 17.5, 1.75)), ('max', (140.0, 12.0, 2.0))],
)
def test_min_and_max_mass_resolve_to_table_masses(mass, expected):
    # Each FL is halfway between the first two FL points of its curve.
    fl = 50.0 if mass == 'min' else 75.0
    assert_performance(climb_table().interpolate(state(fl, mass)), *expected)


def test_max_mass_above_the_heavy_ceiling_resolves_to_the_light_mass():
    # The 80000 kg curve stops at FL150, so at FL180 only 60000 kg is valid.
    # On that curve, FL180 is 0.8 of the way from FL100 to FL200.
    assert_performance(climb_table().interpolate(state(180.0, 'max')), 190.0, 11.0, 1.1)


def test_max_mass_resolves_at_the_query_fl_and_mach():
    # At FL350, the 80000 kg Mach sweep starts at 0.75, so at Mach 0.72 the
    # FL325 query can only use 60000 kg. On that mass, Mach 0.72 is 0.2 of
    # the way from 0.70 to 0.80. FL325 is halfway between FL300 and FL350.
    perf = cruise_table().interpolate(state(325.0, 'max'), mach=0.72)

    assert_performance(perf, 0.5 * 216.0 + 0.5 * 211.0, 0.0, 0.5 * 1.08 + 0.5 * 0.98)


@pytest.mark.parametrize(
    ('table', 'fl', 'extra_coords', 'message'),
    [
        pytest.param(
            climb_table, 250.0, {}, 'climb table: no valid mass for fl=250.0', id='fl'
        ),
        pytest.param(
            cruise_table,
            325.0,
            {'mach': 0.9},
            'cruise table: no valid mass for fl=325.0, mach=0.9',
            id='mach',
        ),
    ],
)
@pytest.mark.parametrize('mass', ['min', 'max'])
def test_min_or_max_mass_with_no_valid_mass_raises(
    table, fl, extra_coords, message, mass
):
    with raises_message(message):
        table().interpolate(state(fl, mass), **extra_coords)


def test_cruise_without_mach_raises():
    with raises_message("cruise performance needs coordinates ['mach'], got []"):
        cruise_table().interpolate(state(300.0, 60000.0))


def test_climb_with_mach_raises():
    with raises_message("climb performance needs coordinates [], got ['mach']"):
        climb_table().interpolate(state(100.0, 60000.0), mach=0.78)


def test_duplicate_key_raises():
    rows = [*CLIMB_ROWS, [100.0, 151.0, 15.0, 1.5, 60000.0]]
    with pytest.raises(ValueError, match='duplicate'):
        climb_table(rows=rows)


def test_nan_output_raises():
    rows = [list(row) for row in CLIMB_ROWS]
    rows[2][CLIMB_COLS.index('fuel_flow')] = math.nan
    with pytest.raises(ValueError, match='NaN'):
        climb_table(rows=rows)


def test_negative_climb_rocd_raises():
    rows = [*CLIMB_ROWS, [250.0, 210.0, -0.1, 1.0, 60000.0]]
    with pytest.raises(ValueError, match='climb'):
        climb_table(rows=rows)


def test_climb_accepts_zero_rocd_at_the_ceiling():
    rows = [*CLIMB_ROWS, [250.0, 210.0, 0.0, 1.0, 60000.0]]
    assert climb_table(rows=rows).interpolate(state(250.0, 60000.0)).rate_of_climb == 0


def test_nonzero_cruise_rocd_raises():
    rows = [list(row) for row in CRUISE_ROWS]
    rows[3][4] = 1.0e-3
    with pytest.raises(ValueError, match='cruise'):
        cruise_table(rows=rows)


@pytest.mark.parametrize('rocd', [0.0, 1.0])
def test_nonnegative_descent_rocd_raises(rocd):
    rows = [*DESCENT_ROWS, [300.0, 200.0, rocd, 0.2, 70000.0]]
    ptin = PerformanceTableInput(cols=CLIMB_COLS, data=rows)
    with pytest.raises(ValueError, match='descent'):
        RaggedPerformanceTable.from_input(ptin, ROCDFilter.NEGATIVE, CLIMB_AXES)


def test_flight_level_from_metres_snaps_to_a_climb_ceiling():
    # FL 34.5 converts to metres and back as 34.50000000000001, which is above
    # the ceiling of the lighter mass.
    fl = 34.5
    assert (fl / METERS_TO_FL) * METERS_TO_FL > fl

    rows = [
        [0.0, 100.0, 20.0, 2.0, 60000.0],
        [fl, 150.0, 15.0, 1.5, 60000.0],
        [0.0, 110.0, 16.0, 2.4, 80000.0],
        [50.0, 170.0, 8.0, 1.6, 80000.0],
    ]
    perf = climb_table(rows=rows).interpolate(state(fl, 60000.0))
    assert perf == Performance(150.0, 15.0, 1.5)


def test_flight_level_from_metres_snaps_to_a_cruise_node():
    # FL 330 converts to metres and back as 330.00000000000006. The FL 340
    # cell starts at a higher Mach, so it must not be used.
    fl = 330.0
    assert (fl / METERS_TO_FL) * METERS_TO_FL > fl

    rows = [
        [fl, 60000.0, 0.40, 130.0, 0.0, 1.0],
        [fl, 60000.0, 0.50, 160.0, 0.0, 1.2],
        [340.0, 60000.0, 0.45, 145.0, 0.0, 1.1],
        [340.0, 60000.0, 0.50, 160.0, 0.0, 1.2],
    ]
    perf = cruise_table(rows=rows).interpolate(state(fl, 60000.0), mach=0.40)
    assert perf == Performance(130.0, 0.0, 1.0)
