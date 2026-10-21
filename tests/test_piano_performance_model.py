"""Tests for `build_piano_model` and `PianoPerformanceModel`.

The fixtures in `tests/data/performance/piano` are dummy data, so
these tests assert that the built model round-trips through the performance
model loader and returns its tabulated values, but do not check that its
numbers are physically plausible.
"""

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
    CruiseMach,
    Performance,
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


def test_operating_empty_mass_survives_the_round_trip(build):
    """PIANO exports do not contain an operating empty mass, so the builder
    takes one and every model file states it."""
    model = build()
    assert model.operating_empty_mass_kg == OPERATING_EMPTY_MASS
    assert model.empty_mass == OPERATING_EMPTY_MASS


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


def test_no_cruise_speeds_are_written(build):
    """PIANO sweeps many cruise Mach numbers, so the model states none."""
    model = build()
    assert model.speeds.cruise is None
    assert model.speeds.climb.mach == pytest.approx(0.750)
    assert model.speeds.descent.mach == pytest.approx(0.750)


def test_name_and_altitude_overrides(build):
    model = build(aircraft_name='other_airplane', maximum_altitude_ft=40000)
    assert model.aircraft_name == 'other_airplane'
    assert model.maximum_altitude_ft == 40000


def test_isa_offset_comes_from_the_exports(build, piano_data):
    model = build()
    assert model.isa_offset == piano_data.isa_offset


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
def test_evaluate_at_every_climb_or_descent_row_returns_the_tabulated_values(
    build, table_name, rules
):
    model = build()

    for row in _rows(getattr(model, table_name)):
        _assert_tabulated(model.evaluate(_state(row), rules), row)


def test_evaluate_at_every_cruise_row_returns_the_tabulated_values(build):
    model = build()

    for row in _rows(model.cruise_flight_performance):
        _assert_tabulated(model.evaluate(_state(row), CruiseMach(row['mach'])), row)


def test_cruise_between_two_mach_points_is_between_the_tabulated_values(build):
    # Interpolation is already tested in test_ragged_interpolator.py but simple
    # check here as well
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

    perf = model.evaluate(_state(low), CruiseMach((low['mach'] + high['mach']) / 2))

    # At the midpoint Mach, linear interpolation gives the mean of the rows.
    for attr, col in (('true_airspeed', 'tas'), ('fuel_flow', 'fuel_flow')):
        mean = (low[col] + high[col]) / 2
        assert getattr(perf, attr) == pytest.approx(mean, rel=1e-12)


def test_simple_cruise_rules_are_refused_with_a_pointer_to_cruise_mach(build):
    model = build()
    row = _rows(model.cruise_flight_performance)[0]

    with pytest.raises(ValueError, match='CruiseMach'):
        model.evaluate(_state(row), SimpleFlightRules.CRUISE)


def test_altitude_above_the_climb_table_is_refused(build):
    model = build()
    top_fl = max(model.climb_flight_performance.column('fl'))
    state = AircraftState(altitude=(top_fl + 10) / METERS_TO_FL, aircraft_mass='min')

    with pytest.raises(ValueError, match='climb table: fl='):
        model.evaluate(state, SimpleFlightRules.CLIMB)
