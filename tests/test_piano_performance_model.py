"""Tests for `build_piano_model`.

The fixtures in `tests/data/performance/piano` are dummy data, so
these tests assert that the built model round-trips through the performance
model loader, but do not check that its numbers are physically plausible.
"""

import pytest

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


@pytest.fixture
def build(tmp_path, piano_data, lto):
    """Build a PIANO model and round-trip it through the model loader."""

    def _build(name: str = 'piano.toml', **overrides) -> PianoPerformanceModel:
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
        return PerformanceModel.load(out_file)

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


def test_empty_mass_requires_an_operating_empty_mass(build):
    """PIANO exports do not contain an operating empty mass, so the field is
    written only when it is manually supplied. An omitted field must load back
    as None rather than failing validation."""
    model = build()
    assert model.operating_empty_mass_kg is None
    with pytest.raises(NotImplementedError, match='operating_empty_mass_kg'):
        model.empty_mass

    with_oew = build('piano_oew.toml', operating_empty_mass=37100)
    assert with_oew.empty_mass == 37100.0


def test_cruise_speeds_written_only_with_cruise_mach(build):
    """PIANO sweeps many cruise Mach numbers, so the model states one only
    when the caller picks one."""
    without = build()
    assert without.speeds.cruise is None
    assert without.speeds.climb.mach == pytest.approx(0.750)
    assert without.speeds.descent.mach == pytest.approx(0.750)

    with_mach = build('piano_mach.toml', cruise_mach=0.78)
    assert with_mach.speeds.cruise is not None
    assert with_mach.speeds.cruise.mach == pytest.approx(0.78)
    # PIANO's cruise table states no CAS schedule.
    assert with_mach.speeds.cruise.cas_low is None
    assert with_mach.speeds.cruise.cas_high is None


def test_name_and_altitude_overrides(build):
    model = build(aircraft_name='other_airplane', maximum_altitude_ft=40000)
    assert model.aircraft_name == 'other_airplane'
    assert model.maximum_altitude_ft == 40000


def test_isa_offset_comes_from_the_exports(build, piano_data):
    model = build()
    assert model.isa_offset == piano_data.isa_offset
