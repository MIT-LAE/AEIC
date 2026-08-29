"""Tests for `build_legacy_model` and the BADA PTF table assembly behind it.

How a model is rendered as TOML is covered in `test_model_writer.py`.
"""

import pytest

from AEIC.parsers.ptf_reader import (
    ClimbPhaseData,
    CruisePhaseData,
    DescentPhaseData,
    PTFData,
)
from AEIC.performance.model_builder import (
    build_legacy_model,
    build_performance_table,
    write_performance_model,
)
from AEIC.performance.models import LegacyPerformanceModel, PerformanceModel
from AEIC.performance.types import SpeedData, Speeds

# Column positions in the table `build_performance_table` returns.
FL, MASS, TAS, ROCD, FUEL_FLOW = range(5)


def make_ptf(low_mass=50000, nominal_mass=60000, high_mass=70000) -> PTFData:
    """A two-flight-level PTF with a distinct value in every column.

    Values are chosen so a row can be traced back to the field it came from,
    rather than for physical plausibility."""
    speed = SpeedData(mach=0.78)
    return PTFData(
        aircraft_type='TEST',
        temperature_reference='ISA +0',
        maximum_altitude_ft=39000,
        low_mass=low_mass,
        nominal_mass=nominal_mass,
        high_mass=high_mass,
        climb=[
            ClimbPhaseData(
                fl=fl,
                tas=200.0 + fl,
                rocd_low=10.0 + fl,
                rocd_nom=8.0 + fl,
                rocd_high=6.0 + fl,
                fuel_flow_nom=0.5 + fl,
            )
            for fl in (100, 200)
        ],
        cruise=[
            CruisePhaseData(
                fl=fl,
                tas=200.0 + fl,
                fuel_flow_low=0.3 + fl,
                fuel_flow_nom=0.4 + fl,
                fuel_flow_high=0.5 + fl,
            )
            for fl in (100, 200)
        ],
        descent=[
            DescentPhaseData(
                fl=fl, tas=200.0 + fl, rocd_nom=-7.0 - fl, fuel_flow_nom=0.2 + fl
            )
            for fl in (100, 200)
        ],
        speeds=Speeds(climb=speed, cruise=speed, descent=speed),
    )


def rows_at(table: dict, mass: float) -> list[list[float]]:
    return [row for row in table['data'] if row[MASS] == mass]


def test_climb_rows_carry_the_rate_of_climb_for_their_own_mass():
    """Climb rate is tabulated per mass, but fuel flow is only given at the
    nominal mass, so every row takes the nominal value."""
    ptf = make_ptf()
    table = build_performance_table(ptf, 'climb')

    assert table['cols'] == ['fl', 'mass', 'tas', 'rocd', 'fuel_flow']
    assert rows_at(table, ptf.low_mass)[0] == [100, ptf.low_mass, 300.0, 110.0, 100.5]
    assert rows_at(table, ptf.nominal_mass)[0] == [
        100,
        ptf.nominal_mass,
        300.0,
        108.0,
        100.5,
    ]
    assert rows_at(table, ptf.high_mass)[0] == [100, ptf.high_mass, 300.0, 106.0, 100.5]


def test_cruise_rows_carry_the_fuel_flow_for_their_own_mass():
    """Cruise holds altitude, so every row has a zero rate of climb."""
    ptf = make_ptf()
    table = build_performance_table(ptf, 'cruise')

    assert rows_at(table, ptf.low_mass)[0] == [100, ptf.low_mass, 300.0, 0.0, 100.3]
    assert rows_at(table, ptf.nominal_mass)[0] == [
        100,
        ptf.nominal_mass,
        300.0,
        0.0,
        100.4,
    ]
    assert rows_at(table, ptf.high_mass)[0] == [100, ptf.high_mass, 300.0, 0.0, 100.5]


def test_descent_rows_exist_only_at_the_nominal_mass():
    """The PTF file tabulates descent at one mass only."""
    ptf = make_ptf()
    table = build_performance_table(ptf, 'descent')

    assert {row[MASS] for row in table['data']} == {ptf.nominal_mass}
    assert table['data'][0] == [100, ptf.nominal_mass, 300.0, -107.0, 100.2]


@pytest.mark.parametrize('phase', ['climb', 'cruise'])
def test_high_mass_rows_dropped_when_they_duplicate_the_nominal_mass(phase):
    """A PTF file may state the same value for the high and nominal masses.
    Emitting both would put two rows at one (flight level, mass) pair, which
    the performance table interpolator rejects."""
    ptf = make_ptf(nominal_mass=60000, high_mass=60000)
    masses = {row[MASS] for row in build_performance_table(ptf, phase)['data']}
    assert masses == {ptf.low_mass, ptf.nominal_mass}

    distinct = build_performance_table(make_ptf(), phase)['data']
    assert {row[MASS] for row in distinct} == {50000, 60000, 70000}


def test_rows_sort_by_mass_then_flight_level():
    """The performance table interpolator needs a dense, regularly ordered
    (flight level, mass) grid."""
    table = build_performance_table(make_ptf(), 'climb')
    keys = [(row[MASS], row[FL]) for row in table['data']]
    assert keys == sorted(keys)
    assert keys == [
        (50000, 100),
        (50000, 200),
        (60000, 100),
        (60000, 200),
        (70000, 100),
        (70000, 200),
    ]


def test_built_model_round_trips_through_the_loader(tmp_path, ptf_data, lto):
    model = build_legacy_model(
        ptf_data,
        lto,
        aircraft_class='narrow',
        number_of_engines=2,
        maximum_payload=22422,
        apu_name='APU 131-9',
    )
    out_file = tmp_path / 'legacy.toml'
    write_performance_model(out_file, model)
    loaded = PerformanceModel.load(out_file)

    assert isinstance(loaded, LegacyPerformanceModel)
    assert loaded.aircraft_name == ptf_data.aircraft_type
    assert loaded.aircraft_class == 'narrow'
    assert loaded.isa_offset == ptf_data.isa_offset
    assert loaded.maximum_altitude_ft == ptf_data.maximum_altitude_ft
    assert loaded.maximum_payload_kg == 22422
    assert loaded.number_of_engines == 2
    assert loaded.apu is not None
    assert loaded.lto_performance == lto


def test_speeds_and_altitude_come_from_the_ptf_file(ptf_data, lto):
    model = build_legacy_model(
        ptf_data,
        lto,
        aircraft_class='narrow',
        number_of_engines=2,
        maximum_payload=22422,
    )
    assert model.speeds == ptf_data.speeds
    assert model.apu_name is None
