"""Tests for `build_piano_model` and `PianoPerformanceModel`.

The fixtures in `tests/data/performance/piano` are dummy data, so
these tests assert that the built model round-trips through the performance
model loader and returns its tabulated values, but do not check that its
numbers are physically plausible.
"""

import math
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from AEIC.parsers.piano_reader.climb_reader import CLIMB_COLS
from AEIC.parsers.piano_reader.cruise_reader import (
    CRUISE_COLS,
    CRUISE_REFERENCE_MACH_COLS,
)
from AEIC.parsers.piano_reader.descent_reader import (
    DESCENT_COLS,
    DESCENT_IDLE_THRUST_COLS,
)
from AEIC.performance.model_builder import build_piano_model, write_performance_model
from AEIC.performance.models import PerformanceModel, PianoPerformanceModel
from AEIC.performance.types import (
    AircraftState,
    Performance,
    PianoConstantMachCruise,
    SimpleFlightRules,
    TableInput,
)
from AEIC.units import METERS_TO_FL, POUNDS_PER_HOUR_TO_KG_PER_S

OPERATING_EMPTY_MASS = 37100
HOLD_FUEL_FLOW_LBPERH = 5000.0


@pytest.fixture
def write_model(tmp_path, piano_data, lto):
    """Write a PIANO model file and return the file it was written to."""

    def _write(name: str = 'piano.toml', **overrides) -> Path:
        overrides.setdefault('operating_empty_mass', OPERATING_EMPTY_MASS)
        overrides.setdefault('hold_fuel_flow_lbperh', HOLD_FUEL_FLOW_LBPERH)
        model = build_piano_model(
            piano_data,
            lto,
            aircraft_class='narrow',
            number_of_engines=2,
            maximum_payload=18000,
            apu_name='APU 131-9',
            **overrides,
        )
        out_file = tmp_path / name
        write_performance_model(out_file, model)
        return out_file

    return _write


@pytest.fixture
def build(write_model):
    """Build a PIANO model and round-trip it through the model loader."""

    def _build(name: str = 'piano.toml', **overrides) -> PianoPerformanceModel:
        return PerformanceModel.load(write_model(name, **overrides))

    return _build


def test_written_file_loads_as_a_piano_model(build):
    model = build()
    assert isinstance(model, PianoPerformanceModel)
    assert model.aircraft_name == 'some_airplane, some_engine'
    assert model.aircraft_class == 'narrow'
    assert model.maximum_altitude_ft == 41100
    assert model.number_of_engines == 2
    assert model.apu is not None
    assert model.apu.name == 'APU 131-9'


def test_every_emitted_column_survives_the_round_trip(build):
    model = build()
    assert model.climb_flight_performance.cols == CLIMB_COLS
    assert model.cruise_flight_performance.cols == CRUISE_COLS
    assert model.descent_flight_performance.cols == DESCENT_COLS
    assert model.cruise_reference_mach.cols == CRUISE_REFERENCE_MACH_COLS
    assert model.descent_idle_thrust.cols == DESCENT_IDLE_THRUST_COLS


def test_loading_fails_without_an_operating_empty_mass(write_model):
    """A file that states no operating empty mass is refused, rather than
    loading a model whose empty mass is undefined."""
    out_file = write_model()
    out_file.write_text(
        ''.join(
            line
            for line in out_file.read_text().splitlines(keepends=True)
            if not line.startswith('operating_empty_mass_kg')
        )
    )

    with pytest.raises(ValidationError, match='PIANO exports do not contain'):
        PerformanceModel.load(out_file)


def test_hold_fuel_flow_survives_the_round_trip_in_kg_per_s(build):
    """The builder takes the hold fuel flow in lb/h, and every model file
    states it in kg/s."""
    model = build()
    assert model.hold_flight_performance.fuel_flow == pytest.approx(
        HOLD_FUEL_FLOW_LBPERH * POUNDS_PER_HOUR_TO_KG_PER_S
    )


def test_loading_fails_without_a_hold_flight_performance(write_model):
    """A file that states no hold fuel flow is refused, and the error names
    the option that supplies it."""
    out_file = write_model()
    text = out_file.read_text()
    out_file.write_text(text[: text.index('[hold_flight_performance]')])

    with pytest.raises(ValidationError, match='--hold-fuel-flow-lbperh'):
        PerformanceModel.load(out_file)


def test_no_cruise_speeds_are_written(write_model):
    """PIANO sweeps many cruise Mach numbers, so the performance model
    should not have a dedicated mach entry. Checks that the file and loaded
    model do not have a cruise speed section/attribute."""
    out_file = write_model()
    with open(out_file, 'rb') as fp:
        assert set(tomllib.load(fp)['speeds']) == {'climb', 'descent'}

    model = PerformanceModel.load(out_file)
    assert model.speeds.cruise is None
    assert model.speeds.climb.mach == pytest.approx(0.750)
    assert model.speeds.descent.mach == pytest.approx(0.750)


def _rows(table: TableInput) -> list[dict[str, float]]:
    """Rows of a table input, keyed by column name."""
    return [dict(zip(table.cols, row)) for row in table.data]


def _state(row: dict[str, float]) -> AircraftState:
    return AircraftState(altitude=row['fl'] / METERS_TO_FL, aircraft_mass=row['mass'])


def _assert_tabulated(perf: Performance, row: dict[str, float]):
    assert perf.true_airspeed == pytest.approx(row['tas'], rel=1e-12)
    assert perf.rate_of_climb == pytest.approx(row['rocd'], rel=1e-12, abs=1e-12)
    assert perf.fuel_flow == pytest.approx(row['fuel_flow'], rel=1e-12)


@pytest.mark.parametrize(
    'table_name, rules',
    [
        ('climb_flight_performance', SimpleFlightRules.CLIMB),
        ('descent_flight_performance', SimpleFlightRules.DESCEND),
    ],
)
def test_evaluate_at_the_current_mass_of_a_block_row_returns_the_row(
    build, table_name, rules
):
    model = build()

    # The table mass is the initial mass of the block. The aircraft at a row
    # has burned the row's fuel, so it is that much lighter.
    for row in _rows(getattr(model, table_name)):
        state = AircraftState(
            altitude=row['fl'] / METERS_TO_FL, aircraft_mass=row['mass'] - row['burn']
        )
        _assert_tabulated(model.evaluate(state, rules), row)


# The fixture climb blocks repeat the same rows, so their values do not change
# with mass. These synthetic tables have one block per initial mass, 60000 and
# 80000 kg, with values that change with the initial mass. They hold only the
# columns that the model needs.
SYNTHETIC_COLS = ['fl', 'mass', 'tas', 'rocd', 'fuel_flow', 'burn']

SYNTHETIC_CLIMB_ROWS = [
    [0.0, 60000.0, 100.0, 20.0, 2.0, 0.0],
    [100.0, 60000.0, 150.0, 15.0, 1.5, 500.0],
    [0.0, 80000.0, 110.0, 16.0, 2.4, 0.0],
    [100.0, 80000.0, 170.0, 8.0, 1.6, 800.0],
]

# A descent starts at the top, so its burn grows as the flight level falls.
SYNTHETIC_DESCENT_ROWS = [
    [0.0, 60000.0, 120.0, -10.0, 0.3, 300.0],
    [100.0, 60000.0, 180.0, -12.0, 0.2, 0.0],
    [0.0, 80000.0, 130.0, -14.0, 0.5, 400.0],
    [100.0, 80000.0, 200.0, -16.0, 0.4, 0.0],
]


@pytest.mark.parametrize(
    'table_name, rows, rules, fl, mass, expected',
    [
        # At FL100 the climbs burned 500 and 800 kg, so the current masses are
        # 59500 and 79200 kg. 64425 kg is 0.25 of the way between them.
        pytest.param(
            'climb_flight_performance',
            SYNTHETIC_CLIMB_ROWS,
            SimpleFlightRules.CLIMB,
            100.0,
            64425.0,
            (
                0.75 * 150.0 + 0.25 * 170.0,
                0.75 * 15.0 + 0.25 * 8.0,
                0.75 * 1.5 + 0.25 * 1.6,
            ),
            id='climb',
        ),
        # At FL0 the descents burned 300 and 400 kg, so the current masses are
        # 59700 and 79600 kg. 64675 kg is 0.25 of the way between them.
        pytest.param(
            'descent_flight_performance',
            SYNTHETIC_DESCENT_ROWS,
            SimpleFlightRules.DESCEND,
            0.0,
            64675.0,
            (
                0.75 * 120.0 + 0.25 * 130.0,
                0.75 * -10.0 + 0.25 * -14.0,
                0.75 * 0.3 + 0.25 * 0.5,
            ),
            id='descent',
        ),
    ],
)
def test_evaluate_between_two_blocks_interpolates_at_the_initial_mass(
    piano_model, table_name, rows, rules, fl, mass, expected
):
    """The current mass is 0.25 of the way between the current masses of the
    two blocks, so the initial mass is 65000 kg. The values are 0.25 of the
    way from the 60000 kg row to the 80000 kg row. A query at the current
    mass, without the conversion to the initial mass, gives other values."""
    data = piano_model.model_dump()
    data[table_name] = {'cols': SYNTHETIC_COLS, 'data': rows}
    model = PianoPerformanceModel.model_validate(data)

    state = AircraftState(altitude=fl / METERS_TO_FL, aircraft_mass=mass)
    perf = model.evaluate(state, rules)

    tas, rocd, fuel_flow = expected
    assert perf.true_airspeed == pytest.approx(tas, rel=1e-12)
    assert perf.rate_of_climb == pytest.approx(rocd, rel=1e-12)
    assert perf.fuel_flow == pytest.approx(fuel_flow, rel=1e-12)


def test_evaluate_at_every_cruise_row_returns_the_tabulated_values(build):
    model = build()

    # Some fixture rows have a negative climb potential, so turn off the check.
    for row in _rows(model.cruise_flight_performance):
        rules = PianoConstantMachCruise(row['mach'], min_rocd_mcl_mach=-math.inf)
        _assert_tabulated(model.evaluate(_state(row), rules), row)


def test_cruise_between_two_mach_points_interpolates_linearly(build):
    # Interpolation is already tested in test_ragged_interpolator.py. This
    # checks that the model passes the Mach number to the cruise table.
    model = build()
    rows = sorted(
        _rows(model.cruise_flight_performance),
        key=lambda r: (r['fl'], r['mass'], r['mach']),
    )
    # The fixture values are mostly constant in Mach. Use two neighbouring
    # Mach points of one cell where the fuel flow differs.
    low, high = next(
        (a, b)
        for a, b in zip(rows, rows[1:])
        if (a['fl'], a['mass']) == (b['fl'], b['mass'])
        and a['fuel_flow'] != b['fuel_flow']
    )

    # Use a Mach 0.25 of the way from the low row, not the midpoint. At the
    # midpoint, a swap of the two interpolation weights gives the same value.
    mach = 0.75 * low['mach'] + 0.25 * high['mach']
    rules = PianoConstantMachCruise(mach, min_rocd_mcl_mach=-math.inf)
    perf = model.evaluate(_state(low), rules)

    for attr, col in (('true_airspeed', 'tas'), ('fuel_flow', 'fuel_flow')):
        expected = 0.75 * low[col] + 0.25 * high[col]
        assert getattr(perf, attr) == pytest.approx(expected, rel=1e-12)


def test_cruise_without_climb_potential_is_refused(build):
    model = build()
    # Get first row with negative climb potential
    row = next(
        r for r in _rows(model.cruise_flight_performance) if r['rocd_mcl_fix_mach'] < 0
    )

    # By default we require that MCL > 0 so should throw
    with pytest.raises(ValueError, match='rate of climb at MCL thrust'):
        model.evaluate(_state(row), PianoConstantMachCruise(row['mach']))


def test_cruise_climb_potential_minimum_comes_from_the_rules(build):
    model = build()
    # Get first row with positive climb potential
    row = next(
        r for r in _rows(model.cruise_flight_performance) if r['rocd_mcl_fix_mach'] > 0
    )

    model.evaluate(_state(row), PianoConstantMachCruise(row['mach']))
    # Make minimum +1 of whatever we got, should throw
    rules = PianoConstantMachCruise(
        row['mach'], min_rocd_mcl_mach=row['rocd_mcl_fix_mach'] + 1.0
    )
    with pytest.raises(ValueError, match='rate of climb at MCL thrust'):
        model.evaluate(_state(row), rules)


def test_simple_cruise_rules_are_refused_with_a_pointer_to_cruise_mach(build):
    model = build()
    row = _rows(model.cruise_flight_performance)[0]

    with pytest.raises(ValueError, match='PianoConstantMachCruise'):
        model.evaluate(_state(row), SimpleFlightRules.CRUISE)


def test_altitude_above_the_climb_table_is_refused(build):
    model = build()
    top_fl = max(model.climb_flight_performance.column('fl'))
    state = AircraftState(
        altitude=(top_fl + 10) / METERS_TO_FL,
        aircraft_mass=min(model.climb_flight_performance.column('mass')),
    )

    with pytest.raises(
        ValueError, match='climb table: no tabulated initial masses reaches fl='
    ):
        model.evaluate(state, SimpleFlightRules.CLIMB)
