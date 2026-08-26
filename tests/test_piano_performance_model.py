"""End-to-end tests for the `make-performance-model piano` subcommand.

The fixtures in `tests/data/performance/piano` are anonymized dummy data, so
these tests assert that the written file round-trips through the performance
model loader, never that its numbers are physically plausible.
"""

import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from AEIC.cli import cli
from AEIC.config import Config
from AEIC.parsers.piano_reader import (
    CLIMB_COLS,
    CRUISE_COLS,
    CRUISE_REFERENCE_MACH_COLS,
    DESCENT_COLS,
    DESCENT_IDLE_THRUST_COLS,
)
from AEIC.performance.models import PerformanceModel, PianoPerformanceModel

TEST_DATA_DIR = (Path(__file__).parent / 'data').resolve()
PIANO_DIR = TEST_DATA_DIR / 'performance' / 'piano'

CLIMB_MASSES_KG = '54000,50000,46000,42000'


# Disable default configuration loading fixture just for this file. The
# `make-performance-model` subcommands call `Config.load()` themselves, which
# fails if the global fixture has already initialized the singleton.
@pytest.fixture(autouse=True)
def default_config(request):
    # Accept `request` and assert no `config_updates` marker is present —
    # without this argument, such a marker on a test in this file would be
    # silently dropped instead of running the global fixture's overlay logic.
    assert request.node.get_closest_marker('config_updates') is None, (
        'tests/test_piano_performance_model.py shadows the global '
        'default_config fixture, so the @pytest.mark.config_updates marker '
        'has no effect here'
    )
    yield None
    Config.reset()


def make_model(out_file: Path, *extra_args: str):
    """Run the piano subcommand and return the click result.

    The subcommand calls `Config.load()`, which refuses to run twice, so
    clear the singleton first.
    """
    Config.reset()
    return CliRunner().invoke(
        cli,
        [
            'make-performance-model',
            '--output-file',
            str(out_file),
            'piano',
            '--lto-source',
            'edb',
            '--engine-file',
            'engines/sample_edb.xlsx',
            '--engine-uid',
            '01P11CM121',
            '--cruise-file',
            str(PIANO_DIR / 'cruise.txt'),
            '--climb-file',
            str(PIANO_DIR / 'climb.txt'),
            '--descent-file',
            str(PIANO_DIR / 'descent.txt'),
            '--climb-masses',
            CLIMB_MASSES_KG,
            '--aircraft-class',
            'narrow',
            '--number-of-engines',
            '2',
            '--maximum-payload',
            '18000',
            '--apu-name',
            'APU 131-9',
            *extra_args,
        ],
    )


def load_model(out_file: Path, *extra_args: str) -> PianoPerformanceModel:
    result = make_model(out_file, *extra_args)
    assert result.exit_code == 0, result.output
    Config.reset()
    Config.load()
    return PerformanceModel.load(out_file)


def test_written_file_loads_as_a_piano_model(tmp_path):
    """Regression guard on the model_type discriminator, which was
    Literal['Piano'] and so could never match a lowercased tag."""
    model = load_model(tmp_path / 'piano.toml')
    assert isinstance(model, PianoPerformanceModel)
    assert model.aircraft_name == 'some_airplane, some_engine'
    assert model.aircraft_class == 'narrow'
    assert model.maximum_altitude_ft == 41100
    assert model.number_of_engines == 2
    assert model.apu is not None


def test_every_emitted_column_survives_the_round_trip(tmp_path):
    model = load_model(tmp_path / 'piano.toml')
    assert model.climb_flight_performance.cols == CLIMB_COLS
    assert model.cruise_flight_performance.cols == CRUISE_COLS
    assert model.descent_flight_performance.cols == DESCENT_COLS
    assert model.cruise_reference_mach.cols == CRUISE_REFERENCE_MACH_COLS
    assert model.descent_idle_thrust.cols == DESCENT_IDLE_THRUST_COLS


def test_maximum_mass_spans_the_three_phase_tables(tmp_path):
    model = load_model(tmp_path / 'piano.toml')
    assert model.maximum_mass == max(
        max(model.climb_flight_performance.column('mass')),
        max(model.cruise_flight_performance.column('mass')),
        max(model.descent_flight_performance.column('mass')),
    )


def test_empty_mass_requires_an_operating_empty_mass(tmp_path):
    """PIANO exports do not contain an operating empty mass."""
    model = load_model(tmp_path / 'piano.toml')
    assert model.operating_empty_mass_kg is None
    with pytest.raises(NotImplementedError, match='operating_empty_mass_kg'):
        model.empty_mass

    with_oew = load_model(
        tmp_path / 'piano_oew.toml', '--operating-empty-mass', '37100'
    )
    assert with_oew.empty_mass == 37100.0


def test_cruise_speeds_written_only_with_cruise_mach(tmp_path):
    """PIANO sweeps many cruise Mach numbers, so the file states one only
    when the user picks one."""
    without = load_model(tmp_path / 'piano.toml')
    assert without.speeds.cruise is None
    assert without.speeds.climb.mach == pytest.approx(0.750)
    assert without.speeds.descent.mach == pytest.approx(0.750)

    with_mach = load_model(tmp_path / 'piano_mach.toml', '--cruise-mach', '0.78')
    assert with_mach.speeds.cruise is not None
    assert with_mach.speeds.cruise.mach == pytest.approx(0.78)
    # PIANO's cruise table states no CAS schedule.
    assert with_mach.speeds.cruise.cas_low is None
    assert with_mach.speeds.cruise.cas_high is None


def test_name_and_altitude_overrides(tmp_path):
    model = load_model(
        tmp_path / 'piano.toml',
        '--aircraft-name',
        'other_airplane',
        '--maximum-altitude-ft',
        '40000',
    )
    assert model.aircraft_name == 'other_airplane'
    assert model.maximum_altitude_ft == 40000


def test_climb_mass_count_mismatch_is_a_usage_error(tmp_path):
    out_file = tmp_path / 'piano.toml'
    result = CliRunner().invoke(
        cli,
        [
            'make-performance-model',
            '--output-file',
            str(out_file),
            'piano',
            '--lto-source',
            'edb',
            '--engine-file',
            'engines/sample_edb.xlsx',
            '--engine-uid',
            '01P11CM121',
            '--cruise-file',
            str(PIANO_DIR / 'cruise.txt'),
            '--climb-file',
            str(PIANO_DIR / 'climb.txt'),
            '--descent-file',
            str(PIANO_DIR / 'descent.txt'),
            '--climb-masses',
            '54000,50000',
            '--aircraft-class',
            'narrow',
            '--number-of-engines',
            '2',
            '--maximum-payload',
            '18000',
        ],
    )
    assert result.exit_code == 2
    assert 'Got 2 climb mass(es) for 4 climb block(s)' in result.output
    assert not out_file.exists()


def test_section_order(tmp_path):
    """Sections are written in the order the subcommand lists them."""
    out_file = tmp_path / 'piano.toml'
    assert make_model(out_file).exit_code == 0
    with open(out_file, 'rb') as fp:
        data = tomllib.load(fp)
    assert [key for key in data if key.endswith(('_performance', '_mach', '_thrust'))][
        -5:
    ] == [
        'climb_flight_performance',
        'cruise_flight_performance',
        'descent_flight_performance',
        'cruise_reference_mach',
        'descent_idle_thrust',
    ]
