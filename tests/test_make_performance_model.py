"""Tests for the `make-performance-model` command line layer.

Building and writing performance models is tested directly in
`test_model_writer.py`, `test_piano_performance_model.py` and
`test_legacy_performance_model.py`. What is left here is what only the command
line does: resolving options into inputs, and reporting bad input as a usage
error.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from AEIC.cli import cli
from AEIC.commands.make_performance_model import (
    check_apu,
    lto_from_edb,
    lto_from_toml,
    parse_climb_masses,
    resolve_lto,
)
from AEIC.config import Config
from AEIC.performance.models.base import LTOPerformanceInput

TEST_DATA_DIR = (Path(__file__).parent / 'data').resolve()

GOLDEN_FILE = TEST_DATA_DIR / 'performance' / 'legacy_golden.toml'
PTF_FILE = TEST_DATA_DIR / 'verification' / 'legacy' / 'legacy_performance.PTF'
PIANO_DIR = TEST_DATA_DIR / 'performance' / 'piano'

ENGINE_FILE = 'engines/sample_edb.xlsx'
ENGINE_UID = '01P11CM121'
THRUST_FRACTIONS = (0.07, 0.30, 0.85, 1.0)


@pytest.fixture
def cli_config(default_config):
    """Hand the configuration singleton back to the command line.

    The `make-performance-model` group callback calls `Config.load()`, which
    refuses to run when a configuration is already loaded. Resetting here keeps
    the global `default_config` fixture's setup, `config_updates` marker
    handling and teardown, which shadowing that fixture would discard."""
    Config.reset()


###########################################
######   End-to-end command output   ######
###########################################


def test_legacy_output_byte_identical(tmp_path, cli_config):
    """The legacy writer's output is frozen. Any diff here is a regression."""
    out_file = tmp_path / 'legacy_performance.toml'

    result = CliRunner().invoke(
        cli,
        [
            'make-performance-model',
            '--output-file',
            str(out_file),
            'legacy',
            '--lto-source',
            'edb',
            '--engine-file',
            ENGINE_FILE,
            '--engine-uid',
            ENGINE_UID,
            '--ptf-file',
            str(PTF_FILE),
            '--aircraft-class',
            'narrow',
            '--number-of-engines',
            '2',
            '--maximum-payload',
            '22422',
            '--apu-name',
            'APU 131-9',
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_file.read_bytes() == GOLDEN_FILE.read_bytes()


def test_bad_input_is_reported_as_a_usage_error(tmp_path, cli_config):
    """Everything the command line calls raises `ValueError` on bad input, and
    one handler per subcommand turns that into a usage error. This pins the
    conversion; the messages themselves are tested at their source."""
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
            ENGINE_FILE,
            '--engine-uid',
            ENGINE_UID,
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


###########################################
######      Option resolution        ######
###########################################


def test_parse_climb_masses():
    assert parse_climb_masses(None) is None
    assert parse_climb_masses('54000,50000.5') == [54000.0, 50000.5]


def test_parse_climb_masses_rejects_a_non_numeric_value():
    with pytest.raises(ValueError, match='comma-separated list of masses'):
        parse_climb_masses('54000,heavy')


def test_check_apu_accepts_a_known_name_and_no_name():
    assert check_apu(None) is None
    assert check_apu('APU 131-9') is None


def test_check_apu_rejects_an_unknown_name():
    with pytest.raises(ValueError, match='not found in APU database'):
        check_apu('APU 000-0')


def test_lto_from_edb_requires_both_engine_options():
    with pytest.raises(ValueError, match='--engine-file and --engine-uid'):
        lto_from_edb(None, ENGINE_UID, THRUST_FRACTIONS)
    with pytest.raises(ValueError, match='--engine-file and --engine-uid'):
        lto_from_edb(ENGINE_FILE, None, THRUST_FRACTIONS)


def test_lto_from_toml_requires_a_file():
    with pytest.raises(ValueError, match='--lto-file must be provided'):
        lto_from_toml(None)


def test_lto_from_toml_reads_an_lto_performance_table():
    """The reader accepts either a whole model file or a bare LTO table."""
    lto = lto_from_toml(GOLDEN_FILE)
    assert lto.ICAO_UID == ENGINE_UID


def test_resolve_lto_rejects_an_unknown_source():
    with pytest.raises(ValueError, match='Unsupported LTO source'):
        resolve_lto('nowhere', None, None, THRUST_FRACTIONS, None)


def test_resolve_lto_from_edb_returns_performance_model_input():
    """The engine file is resolved through the AEIC search path, so a bare
    name works here as it does on the command line."""
    lto = resolve_lto('edb', ENGINE_FILE, ENGINE_UID, THRUST_FRACTIONS, None)
    assert isinstance(lto, LTOPerformanceInput)
    assert lto.source == 'EDB'
    assert lto.ICAO_UID == ENGINE_UID


def test_resolve_lto_from_a_custom_file():
    lto = resolve_lto('custom', None, None, THRUST_FRACTIONS, GOLDEN_FILE)
    assert isinstance(lto, LTOPerformanceInput)
    assert lto.ICAO_UID == ENGINE_UID
